"""
Generate ARIA installer branding images for Inno Setup.
Run once before compiling installer.iss:
    python build_installer_assets.py

Outputs:
    assets/wizard_banner.bmp   — 164x314 sidebar panel (dark gradient + ARIA text)
    assets/wizard_header.bmp   —  55x58  small header logo
    assets/aria.ico            — Multi-size Windows icon
"""

import os
import struct
import math

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)


# ── Minimal BMP writer (no Pillow needed) ────────────────────────────────────

def write_bmp(path: str, width: int, height: int, pixels):
    """
    Write a 24-bit BMP file.
    pixels: list of (R, G, B) tuples, row-major from top-left.
    BMP rows must be padded to 4-byte boundary and stored bottom-up.
    """
    row_size = width * 3
    pad = (4 - row_size % 4) % 4
    padded_row = row_size + pad
    pixel_data_size = padded_row * height
    file_size = 54 + pixel_data_size

    with open(path, "wb") as f:
        # File header (14 bytes)
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", 54))
        # DIB header (40 bytes)
        f.write(struct.pack("<I", 40))
        f.write(struct.pack("<i", width))
        f.write(struct.pack("<i", -height))   # negative = top-down
        f.write(struct.pack("<HH", 1, 24))
        f.write(struct.pack("<I", 0))         # no compression
        f.write(struct.pack("<I", pixel_data_size))
        f.write(struct.pack("<ii", 2835, 2835))
        f.write(struct.pack("<II", 0, 0))
        # Pixel data
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[y * width + x]
                f.write(bytes([b, g, r]))     # BMP stores BGR
            f.write(b"\x00" * pad)

    print(f"[OK] {path}  ({width}x{height})")


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ── Banner: 164 x 314 — dark space gradient + "ARIA" text ───────────────────

def make_banner():
    W, H = 164, 314
    pixels = []

    # Background: deep navy -> dark blue-purple gradient
    TOP    = (10,  15,  40)   # #0A0F28
    BOTTOM = (25,  30,  80)   # #191E50

    for y in range(H):
        t = y / (H - 1)
        base = lerp_color(TOP, BOTTOM, t)

        for x in range(W):
            r, g, b = base

            # Subtle vertical shimmer lines
            if x % 20 == 0:
                r = min(255, r + 10)
                g = min(255, g + 10)
                b = min(255, b + 18)

            # Soft glow in the upper-center area
            cx, cy = W // 2, H // 4
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            glow = max(0, 1 - dist / 90)
            r = min(255, int(r + glow * 15))
            g = min(255, int(g + glow * 20))
            b = min(255, int(b + glow * 60))

            pixels.append((r, g, b))

    # Draw "ARIA" text block (simple pixel font, 5x7 per char)
    FONT_5x7 = {
        'A': ["01110","10001","10001","11111","10001","10001","10001"],
        'R': ["11110","10001","10001","11110","10100","10010","10001"],
        'I': ["11111","00100","00100","00100","00100","00100","11111"],
    }

    def draw_char(pixels, W, char, ox, oy, color, scale=3):
        pattern = FONT_5x7.get(char, [])
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit == '1':
                    for dy in range(scale):
                        for dx in range(scale):
                            px = ox + col * scale + dx
                            py = oy + row * scale + dy
                            if 0 <= px < W and 0 <= py < len(pixels) // W:
                                pixels[py * W + px] = color

    CYAN = (0, 212, 255)
    text_y = 60
    text_x = 20
    for ch in "ARIA":
        draw_char(pixels, W, ch, text_x, text_y, CYAN, scale=3)
        text_x += 5 * 3 + 4  # char_width * scale + spacing

    # Draw thin horizontal accent line below ARIA text
    line_y = text_y + 7 * 3 + 8
    for x in range(16, W - 16):
        t = (x - 16) / (W - 32)
        a = min(1.0, 2 * min(t, 1 - t))          # fade in/out
        r = int(0   + a * CYAN[0])
        g = int(0   + a * CYAN[1])
        b = int(40  + a * (CYAN[2] - 40))
        pixels[line_y * W + x] = (r, g, b)

    # Small tagline dots
    dot_y = line_y + 12
    for i in range(3):
        cx = W // 2 - 12 + i * 12
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx * dx + dy * dy <= 4:
                    px, py = cx + dx, dot_y + dy
                    if 0 <= px < W and 0 <= py < H:
                        pixels[py * W + px] = CYAN

    write_bmp(os.path.join(ASSETS, "wizard_banner.bmp"), W, H, pixels)


# ── Small header logo: 55 x 58 — circle with "A" ────────────────────────────

def make_header():
    W, H = 55, 58
    pixels = []
    BG     = (18, 22, 56)     # dark bg
    CYAN   = (0, 212, 255)
    WHITE  = (220, 230, 255)

    for y in range(H):
        for x in range(W):
            cx, cy = W // 2, H // 2
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < 24:
                if dist > 20:
                    # Ring border
                    t = (dist - 20) / 4
                    c = lerp_color(CYAN, BG, t)
                    pixels.append(c)
                else:
                    # Interior
                    glow = max(0, 1 - dist / 22)
                    r = int(BG[0] + glow * 15)
                    g = int(BG[1] + glow * 20)
                    b = int(BG[2] + glow * 60)
                    pixels.append((r, g, b))
            else:
                pixels.append(BG)

    # Draw "A" in center
    A_PATTERN = [
        " 000 ",
        "0   0",
        "0   0",
        "00000",
        "0   0",
        "0   0",
        "0   0",
    ]
    ox, oy = W // 2 - 8, H // 2 - 10
    for row, bits in enumerate(A_PATTERN):
        for col, bit in enumerate(bits):
            if bit == '0':
                for dy in range(2):
                    for dx in range(2):
                        px, py = ox + col * 3 + dx, oy + row * 2 + dy
                        if 0 <= px < W and 0 <= py < H:
                            pixels[py * W + px] = WHITE

    write_bmp(os.path.join(ASSETS, "wizard_header.bmp"), W, H, pixels)


# ── ICO writer (16x16 and 32x32 embedded) ────────────────────────────────────

def make_icon():
    """Create a minimal multi-size .ico with 16x16 and 32x32 images."""

    def icon_pixels(size):
        px = []
        BG   = (18, 22, 56)
        CYAN = (0, 212, 255)
        for y in range(size):
            for x in range(size):
                cx, cy = size / 2, size / 2
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                r_outer = size * 0.46
                r_inner = size * 0.32
                if dist <= r_outer:
                    if dist > r_inner:
                        t = (dist - r_inner) / (r_outer - r_inner)
                        c = lerp_color(CYAN, BG, t)
                        px.append(c + (255,))
                    else:
                        glow = max(0, 1 - dist / r_inner)
                        b = int(BG[2] + glow * 60)
                        px.append((BG[0], BG[1], b, 255))
                else:
                    px.append((0, 0, 0, 0))  # transparent
        return px

    def bmp_data_for_ico(size, pixels):
        """Return raw BMP data suitable for embedding in ICO (without file header)."""
        row_size = size * 4  # 32-bit BGRA
        data = b""
        # DIB header (40 bytes), height doubled for XOR+AND masks
        data += struct.pack("<I", 40)
        data += struct.pack("<i", size)
        data += struct.pack("<i", size * 2)   # doubled
        data += struct.pack("<HH", 1, 32)
        data += struct.pack("<I", 0)
        data += struct.pack("<I", row_size * size)
        data += struct.pack("<ii", 2835, 2835)
        data += struct.pack("<II", 0, 0)
        # Pixel rows (bottom-up, BGRA)
        for y in range(size - 1, -1, -1):
            for x in range(size):
                r, g, b, a = pixels[y * size + x]
                data += bytes([b, g, r, a])
        # AND mask (all zeros = fully use XOR data)
        mask_row = ((size + 31) // 32) * 4
        data += b"\x00" * (mask_row * size)
        return data

    sizes = [16, 32, 48]
    images = []
    for sz in sizes:
        px = icon_pixels(sz)
        bmp = bmp_data_for_ico(sz, px)
        images.append((sz, bmp))

    # ICO file header
    ico_header = struct.pack("<HHH", 0, 1, len(images))

    # Directory entries
    offset = 6 + len(images) * 16
    entries = b""
    for sz, bmp in images:
        w = 0 if sz >= 256 else sz
        h = 0 if sz >= 256 else sz
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(bmp), offset)
        offset += len(bmp)

    ico_path = os.path.join(ASSETS, "aria.ico")
    with open(ico_path, "wb") as f:
        f.write(ico_header)
        f.write(entries)
        for _, bmp in images:
            f.write(bmp)

    print(f"[OK] {ico_path}  ({', '.join(str(s) for s in sizes)}px)")


if __name__ == "__main__":
    print("Generating ARIA installer assets...")
    make_banner()
    make_header()
    make_icon()
    print("\nDone! Assets written to:", ASSETS)
    print("\nNext steps:")
    print("  1. Open installer.iss in Inno Setup Compiler")
    print("  2. Add '#define ARIA_HAS_ASSETS' at the top of installer.iss")
    print("  3. Press Ctrl+F9 to compile")
    print("  4. Output: dist\\installer\\ARIA_Setup_1.0.0.exe")
