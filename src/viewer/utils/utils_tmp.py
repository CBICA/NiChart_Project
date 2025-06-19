from PIL import Image, ImageDraw

def make_circle_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    size = img.size

    # Create same-size mask with a white circle on black background
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)

    # Apply circular mask to image
    img.putalpha(mask)

    # Save with transparency
    img.save(output_path, format="PNG")

## Example
#make_circle_image("square_logo.png", "circle_logo.png")

def make_scaled_circle_image(input_path, output_path, scale=0.9):
    img = Image.open(input_path).convert("RGBA")
    w, h = img.size

    # Crop to square (center crop)
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))

    # Resize (scale down)
    new_size = int(min_dim * scale)
    img_resized = img.resize((new_size, new_size), Image.LANCZOS)

    # Create final square canvas
    final_img = Image.new("RGBA", (min_dim, min_dim), (0, 0, 0, 0))

    # Center resized image on canvas
    offset = ((min_dim - new_size) // 2, (min_dim - new_size) // 2)
    final_img.paste(img_resized, offset)

    # Create circular mask
    mask = Image.new('L', (min_dim, min_dim), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, min_dim, min_dim), fill=255)

    # Apply mask
    final_img.putalpha(mask)

    # Save result
    final_img.save(output_path, format="PNG")

## Example usage
#make_scaled_circle_image("your_image.png", "circle_output.png", scale=0.85)


