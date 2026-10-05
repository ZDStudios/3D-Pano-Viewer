"""Stitch DJI sphere-panorama shots into a single equirectangular image.

DJI drones write the gimbal yaw/pitch/roll of every shot into XMP metadata,
so we use that as the starting camera orientation, refine it with feature
matching + bundle adjustment, then blend all shots into a 2:1 equirect.
Images without DJI metadata fall back to OpenCV's generic stitcher.
"""
import math
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _log(cb, msg, frac=None):
    if cb:
        cb(msg, frac)


# ---------------------------------------------------------------- metadata
def read_dji_meta(path):
    with open(path, "rb") as fh:
        head = fh.read(256 * 1024)
    tags = dict(re.findall(rb'drone-dji:(\w+)="([^"]*)"', head))
    try:
        yaw = float(tags[b"GimbalYawDegree"])
        pitch = float(tags[b"GimbalPitchDegree"])
        roll = float(tags.get(b"GimbalRollDegree", b"0"))
    except (KeyError, ValueError):
        return None
    return yaw, pitch, roll


def focal_px(path, width, height):
    """Focal length in pixels from the EXIF 35mm-equivalent focal length."""
    f35 = None
    try:
        exif = Image.open(path).getexif().get_ifd(0x8769)
        f35 = exif.get(0xA405)
    except Exception:
        pass
    f35 = float(f35) if f35 else 24.0  # DJI Mini 4 Pro = 24mm equiv
    diag_px = math.hypot(width, height)
    return f35 * diag_px / math.hypot(36, 24)


def cam_to_world(yaw, pitch, roll):
    """Rotation (3x3) mapping OpenCV camera coords (x right, y down, z fwd)
    to world coords (x east, y up, z north)."""
    y, p = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])
    right = np.array([math.cos(y), 0.0, -math.sin(y)])
    up = np.cross(fwd, right)
    R = np.stack([right, -up, fwd], axis=1)
    if roll:
        R = R @ Rotation.from_rotvec([0, 0, math.radians(roll)]).as_matrix()
    return R


# ---------------------------------------------------------------- refine
def _features(img_small):
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=3000)
    kp, des = sift.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des


def _rays(pts, f, cx, cy):
    v = np.column_stack([(pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f, np.ones(len(pts))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def refine_rotations(smalls, Rs, f, cb=None):
    n = len(smalls)
    h, w = smalls[0].shape[:2]
    cx, cy = w / 2, h / 2
    _log(cb, "Detecting features…", 0.15)
    with ThreadPoolExecutor() as ex:
        feats = list(ex.map(_features, smalls))

    fwd = np.array([R[:, 2] for R in Rs])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.dot(fwd[i], fwd[j]) > math.cos(math.radians(70))]

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(pair):
        i, j = pair
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 20 or len(dj) < 20:
            return None
        knn = matcher.knnMatch(di, dj, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 25:
            return None
        a = pi[[m.queryIdx for m in good]]
        b = pj[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
        if H is None:
            return None
        mask = mask.ravel().astype(bool)
        if mask.sum() < 20:
            return None
        a, b = a[mask], b[mask]
        # reject pairs that disagree wildly with the gimbal prior
        ra = _rays(a, f, cx, cy) @ Rs[i].T
        rb = _rays(b, f, cx, cy) @ Rs[j].T
        if np.median(np.linalg.norm(ra - rb, axis=1)) > math.radians(8):
            return None
        sel = np.random.default_rng(i * 1000 + j).permutation(len(a))[:80]
        return i, j, a[sel], b[sel]

    _log(cb, f"Matching {len(pairs)} image pairs…", 0.3)
    with ThreadPoolExecutor() as ex:
        matches = [m for m in ex.map(match, pairs) if m]
    if not matches:
        return Rs, f

    I = np.concatenate([np.full(len(m[2]), m[0]) for m in matches])
    J = np.concatenate([np.full(len(m[2]), m[1]) for m in matches])
    A = np.concatenate([m[2] for m in matches])
    B = np.concatenate([m[3] for m in matches])
    R0 = np.array(Rs)

    def residuals(x):
        d = Rotation.from_rotvec(x[:3 * n].reshape(n, 3)).as_matrix()
        Rn = np.einsum("nij,njk->nik", d, R0)
        fs = f * x[-1]
        ra = np.einsum("nij,nj->ni", Rn[I], _rays(A, fs, cx, cy))
        rb = np.einsum("nij,nj->ni", Rn[J], _rays(B, fs, cx, cy))
        prior = 0.02 * x[:3 * n]  # keep the horizon close to the gimbal values
        return np.concatenate([(ra - rb).ravel() * f, prior * f])

    _log(cb, f"Aligning ({len(A)} matched points)…", 0.4)
    x0 = np.zeros(3 * n + 1)
    x0[-1] = 1.0
    from scipy.sparse import lil_matrix
    m = len(A)
    sp = lil_matrix((3 * m + 3 * n, 3 * n + 1), dtype=np.int8)
    rr = np.arange(3 * m)
    for k in range(3):
        for o in range(3):
            sp[rr, 3 * np.repeat(I, 3) + k] = 1
            sp[rr, 3 * np.repeat(J, 3) + k] = 1
    sp[:3 * m, -1] = 1
    sp[3 * m + np.arange(3 * n), np.arange(3 * n)] = 1
    res = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, max_nfev=40,
                        x_scale="jac", jac_sparsity=sp, method="trf")
    x = res.x
    d = Rotation.from_rotvec(x[:3 * n].reshape(n, 3)).as_matrix()
    Rn = [d[k] @ R0[k] for k in range(n)]
    # Remove any global drift the solver introduced: find the single rotation
    # that best maps the refined cameras back onto the gimbal orientations.
    M = sum(R0[k] @ Rn[k].T for k in range(n))
    U, _, Vt = np.linalg.svd(M)
    G = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return [G @ R for R in Rn], f * x[-1]


# ---------------------------------------------------------------- blend
def _gain_compensation(smalls, Rs, f):
    """Per-image brightness gains from overlap regions (simple, robust)."""
    n = len(smalls)
    h, w = smalls[0].shape[:2]
    W, H = 512, 256
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    means, masks = [], []
    for k in range(n):
        c = dirs @ Rs[k]
        z = c[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (f * c[..., 0] / z + w / 2).astype(np.float32)
            my = (f * c[..., 1] / z + h / 2).astype(np.float32)
        m = (z > 0) & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        mx[~m] = -1
        my[~m] = -1
        warped = cv2.remap(smalls[k], mx, my, cv2.INTER_LINEAR).astype(np.float32)
        means.append(warped.mean(axis=2) + 1.0)
        masks.append(m)
    # Solve log-gains: minimize sum over overlaps (g_i + l_i - g_j - l_j)^2
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ov = masks[i] & masks[j]
            if ov.sum() < 50:
                continue
            diff = np.log(means[i][ov]).mean() - np.log(means[j][ov]).mean()
            r = np.zeros(n)
            r[i], r[j] = 1, -1
            rows.append(r * math.sqrt(ov.sum()))
            rhs.append(-diff * math.sqrt(ov.sum()))
    for i in range(n):
        r = np.zeros(n)
        r[i] = 10.0
        rows.append(r)
        rhs.append(0.0)
    g = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return np.exp(g)


def _voronoi_feed(paths, Rs, f_full, out_w, cb, gains):
    """Blend using nearest-centre seams + multiband to avoid ghosting."""
    out_h = out_w // 2
    n = len(paths)
    blender = cv2.detail_MultiBandBlender(0, 6)
    blender.prepare((0, 0, out_w, out_h))
    lon = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * np.pi

    # first pass at reduced size: compute owner map (image whose centre is
    # closest, among images that cover the pixel)
    sw, sh = out_w // 4, out_h // 4
    lon_s, lat_s = np.meshgrid(lon[::4][:sw], lat[::4][:sh])
    ds = np.stack([np.cos(lat_s) * np.sin(lon_s), np.sin(lat_s), np.cos(lat_s) * np.cos(lon_s)], -1)
    best = np.full((sh, sw), -1, np.int32)
    best_score = np.full((sh, sw), -np.inf, np.float32)
    sizes = []
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            sizes.append(im.size)
    for k in range(n):
        w, h = sizes[k]
        c = ds @ Rs[k]
        z = c[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f_full * c[..., 0] / z
            v = f_full * c[..., 1] / z
        valid = (z > 0) & (np.abs(u) < w / 2 - 2) & (np.abs(v) < h / 2 - 2)
        score = np.where(valid, np.minimum(1 - np.abs(u) / (w / 2), 1 - np.abs(v) / (h / 2)), -np.inf)
        upd = score > best_score
        best[upd] = k
        best_score[upd] = score[upd]
    coverage_s = best >= 0
    best_full = cv2.resize(best, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    for k, path in enumerate(paths):
        _log(cb, f"Blending image {k + 1}/{n}…", 0.55 + 0.4 * k / n)
        own = best_full == k
        if not own.any():
            continue
        # dilate ownership so multiband has overlap to blend across
        own = cv2.dilate(own.astype(np.uint8), np.ones((31, 31), np.uint8))
        ys, xs = np.nonzero(own[:, :])
        y0, y1 = ys.min(), ys.max() + 1
        colmask = own[y0:y1].any(axis=0)
        cols = np.nonzero(colmask)[0]
        gaps = np.diff(cols)
        if cols[0] == 0 and cols[-1] == out_w - 1 and gaps.size and gaps.max() > 1:
            s = np.argmax(gaps)
            ranges = [(0, cols[s] + 1), (cols[s + 1], out_w)]
        else:
            ranges = [(cols[0], cols[-1] + 1)]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if gains is not None:
            img = np.clip(img.astype(np.float32) * gains[k], 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        R = Rs[k].astype(np.float32)
        for x0, x1 in ranges:
            lo, la = np.meshgrid(lon[x0:x1], lat[y0:y1])
            cl = np.cos(la)
            d = np.stack([cl * np.sin(lo), np.sin(la), cl * np.cos(lo)], -1)
            c = d @ R
            z = c[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                mx = f_full * c[..., 0] / z + w / 2
                my = f_full * c[..., 1] / z + h / 2
            valid = (z > 0) & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
            mx = np.where(valid, mx, -1).astype(np.float32)
            my = np.where(valid, my, -1).astype(np.float32)
            warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask = (valid & own[y0:y1, x0:x1].astype(bool)).astype(np.uint8) * 255
            if mask.any():
                blender.feed(warped.astype(np.int16), mask, (int(x0), int(y0)))
    _log(cb, "Finalising panorama…", 0.96)
    result, _ = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)
    coverage = cv2.resize(coverage_s.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return result, coverage


def _fill_holes(pano, coverage):
    """Fill uncovered areas (usually a small cap of sky at the zenith)."""
    holes = (coverage == 0).astype(np.uint8)
    if not holes.any():
        return pano
    h, w = pano.shape[:2]
    # multiband blending darkens a thin rim along the hole edge; include it
    k = max(3, w // 150)
    holes = cv2.dilate(holes, np.ones((k, k), np.uint8))
    # inpaint at low res for speed, then paste back smoothly
    s = 1024 / w
    small = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(holes, (1024, 512), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((5, 5), np.uint8))
    filled = cv2.inpaint(small, hs * 255, 9, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 6)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    m = holes.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), k / 3)[..., None]
    return (pano * (1 - m) + up * m).astype(np.uint8)


# ---------------------------------------------------------------- public
def list_images(folder):
    import os
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."))


def stitch_folder(folder, out_w=8192, cb=None):
    paths = list_images(folder)
    if not paths:
        raise ValueError("No images in folder")

    # Already a stitched equirect?
    if len(paths) == 1:
        img = cv2.imread(paths[0])
        h, w = img.shape[:2]
        if abs(w / h - 2) > 0.05:
            raise ValueError("Single image is not a 2:1 equirectangular panorama")
        return img

    metas = [read_dji_meta(p) for p in paths]
    if any(m is None for m in metas):
        return _opencv_stitch(paths, cb)

    _log(cb, f"Loading {len(paths)} images…", 0.02)
    with Image.open(paths[0]) as im:
        W0, H0 = im.size
    f_full = focal_px(paths[0], W0, H0)
    scale = 1008 / W0

    def load_small(p):
        img = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
        return cv2.resize(img, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)

    with ThreadPoolExecutor() as ex:
        smalls = list(ex.map(load_small, paths))
    Rs = [cam_to_world(*m) for m in metas]
    try:
        Rs, f_small = refine_rotations(smalls, Rs, f_full * scale, cb)
        f_full = f_small / scale
    except Exception as e:  # alignment is best-effort; gimbal data alone is decent
        _log(cb, f"Alignment refinement skipped ({e})", 0.5)

    _log(cb, "Balancing exposure…", 0.5)
    gains = _gain_compensation(smalls, Rs, f_full * scale)
    pano, coverage = _voronoi_feed(paths, Rs, f_full, out_w, cb, gains)
    pano = _fill_holes(pano, coverage)
    _log(cb, "Done", 1.0)
    return pano


def _opencv_stitch(paths, cb):
    _log(cb, "No DJI metadata – using generic stitcher (slower)…", 0.1)
    imgs = [cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_2) for p in paths]
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = st.stitch(imgs)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed (OpenCV status {status})")
    # pad to 2:1 so it can wrap a sphere
    h, w = pano.shape[:2]
    if w / h < 2:
        H = h
        W = 2 * h
        canvas = np.zeros((H, W, 3), np.uint8)
        x0 = (W - w) // 2
        canvas[:, x0:x0 + w] = pano
        pano = canvas
    else:
        H = w // 2
        canvas = np.zeros((H, w, 3), np.uint8)
        y0 = (H - h) // 2
        canvas[y0:y0 + h] = pano
        pano = canvas
    _log(cb, "Done", 1.0)
    return pano


if __name__ == "__main__":
    import sys
    import time
    t = time.time()
    out = stitch_folder(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 8192,
                        cb=lambda m, f: print(f"[{time.time() - t:6.1f}s] {m}"))
    cv2.imwrite(sys.argv[2], out, [cv2.IMWRITE_JPEG_QUALITY, 92])
