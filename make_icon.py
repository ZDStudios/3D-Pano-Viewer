"""Draws icon.ico (globe + orbit arrow + drone) at several sizes."""
import math
from PIL import Image, ImageDraw, ImageFilter

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
# rounded-square background with vertical gradient
bg = Image.new("RGBA", (S, S))
d = ImageDraw.Draw(bg)
for y in range(S):
    t = y / S
    d.line([(0, y), (S, y)], fill=(int(18 + 10 * t), int(40 + 30 * t), int(80 + 40 * t), 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([24, 24, S - 24, S - 24], 210, fill=255)
img.paste(bg, (0, 0), mask)

# globe: sky top half, sand/sea bottom half
cx, cy, r = S // 2, S // 2 + 30, 300
globe = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = ImageDraw.Draw(globe)
for y in range(cy - r, cy + r):
    t = (y - (cy - r)) / (2 * r)
    if t < 0.5:
        c = (int(70 + 80 * t), int(150 + 80 * t), 255)
    elif t < 0.62:
        c = (40, 170, 190)
    else:
        c = (236, 220, 180)
    g.line([(cx - r, y), (cx + r, y)], fill=c + (255,))
gm = Image.new("L", (S, S), 0)
ImageDraw.Draw(gm).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
sd = ImageDraw.Draw(shade)
for i in range(60):
    rr = r - i * 2
    sd.ellipse([cx - rr - 60 + i, cy - rr - 60 + i, cx + rr - 60 + i, cy + rr - 60 + i], outline=(255, 255, 255, 2))
globe = Image.alpha_composite(globe, shade)
# meridians / parallels
lines = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ld = ImageDraw.Draw(lines)
for k in (-0.6, 0, 0.6):
    ry = abs(r * math.sin(math.radians(90 * k))) if k else 0
    ld.ellipse([cx - r, cy - r * 0.32 + k * r * 0.95 - r * 0.32 * 0 , cx + r, cy + r * 0.32 + k * r * 0.95], outline=(255, 255, 255, 90), width=8)
for k in (0.35, 0.75):
    ld.ellipse([cx - r * k, cy - r, cx + r * k, cy + r], outline=(255, 255, 255, 90), width=8)
ld.line([(cx, cy - r), (cx, cy + r)], fill=(255, 255, 255, 90), width=8)
globe = Image.alpha_composite(globe, lines)
img.paste(globe, (0, 0), gm)
ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 230), width=14)

# orbit arrow
od = ImageDraw.Draw(img)
box = [cx - 420, cy - 120, cx + 420, cy + 120]
od.arc(box, 200, 340, fill=(255, 196, 60, 255), width=30)
od.arc(box, 20, 160, fill=(255, 196, 60, 255), width=30)
ang = math.radians(340)
ax, ay = cx + 420 * math.cos(ang), cy + 120 * math.sin(ang)
od.polygon([(ax + 40, ay + 10), (ax - 50, ay - 40), (ax - 20, ay + 50)], fill=(255, 196, 60, 255))

# drone silhouette on top
dx, dy = cx, 170
w = (255, 255, 255, 255)
od.rounded_rectangle([dx - 70, dy - 26, dx + 70, dy + 26], 22, fill=w)
for sx in (-1, 1):
    od.line([(dx + sx * 60, dy), (dx + sx * 150, dy - 30)], fill=w, width=20)
    od.ellipse([dx + sx * 150 - 75, dy - 48, dx + sx * 150 + 75, dy - 12], fill=(255, 255, 255, 200))
od.ellipse([dx - 18, dy + 18, dx + 18, dy + 54], fill=(30, 30, 40, 255), outline=w, width=6)

img.save("icon.png")
img.save("icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
