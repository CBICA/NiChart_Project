from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

def create_circular_pin(image_path, output_path='circular_pin.png'):
    # Load and convert to RGBA
    im = Image.open(image_path).convert("RGBA")
    size = min(im.size)

    # Crop to square
    im = ImageOps.fit(im, (size, size), centering=(0.5, 0.5))

    # Create circular mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Apply mask to create circular image
    circular_im = Image.new("RGBA", (size, size))
    circular_im.paste(im, (0, 0), mask=mask)

    # Add a metallic pin border (silver)
    border_width = int(size * 0.05)
    border_im = Image.new("RGBA", (size + 2 * border_width, size + 2 * border_width), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_im)
    outer = (0, 0, size + 2 * border_width, size + 2 * border_width)
    inner = (border_width, border_width, size + border_width, size + border_width)
    border_draw.ellipse(outer, fill=(192, 192, 192, 255))  # silver rim
    border_draw.ellipse(inner, fill=(0, 0, 0, 0))  # hollow center
    border_im.paste(circular_im, (border_width, border_width), circular_im)

    # Add shadow
    shadow = border_im.filter(ImageFilter.GaussianBlur(radius=8))
    shadow_layer = Image.new("RGBA", border_im.size, (0, 0, 0, 0))
    shadow_layer.paste(shadow, (6, 6), shadow)

    # Merge shadow and pin
    final = Image.alpha_composite(shadow_layer, border_im)

    # Add highlight for 3D effect
    highlight = ImageDraw.Draw(final)
    highlight.ellipse(
        (border_width + 10, border_width + 10, size // 2, size // 2),
        fill=(255, 255, 255, 40)
    )

    # Save
    final.save(output_path)
    print(f"Saved circular pin image to {output_path}")

# Example usage:
create_circular_pin("logo_dlmuse_init.png")
