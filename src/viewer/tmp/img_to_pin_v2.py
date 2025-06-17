from PIL import Image, ImageDraw, ImageFilter
import math

def create_circular_pin_effect(image_path, output_path,
                               radius_percent=90, border_size=10, border_color=(180, 180, 180, 255), # Grey for metal
                               shadow_offset_x=10, shadow_offset_y=10, shadow_blur_radius=15, shadow_opacity=150, # RGBA
                               pin_needle_color=(50, 50, 50, 255), pin_needle_length_ratio=0.8, pin_needle_width=3, pin_needle_angle=20):
    """
    Converts a square image into a circular pin-like image with basic 3D effects.

    Args:
        image_path (str): Path to the input square image.
        output_path (str): Path to save the output pin image.
        radius_percent (int): Percentage of the image's shortest side to use as the pin's radius.
                               e.g., 90 means the circle will take 90% of the image's width/height.
        border_size (int): Size of the border around the circular pin (mimics thickness).
        border_color (tuple): RGBA color of the border (e.g., (180, 180, 180, 255) for light grey).
        shadow_offset_x (int): X-offset for the drop shadow.
        shadow_offset_y (int): Y-offset for the drop shadow.
        shadow_blur_radius (int): Blur radius for the drop shadow.
        shadow_opacity (int): Opacity of the drop shadow (0-255).
        pin_needle_color (tuple): RGBA color of the pin needle.
        pin_needle_length_ratio (float): Length of the pin needle as a ratio of pin diameter (0 to 1).
        pin_needle_width (int): Width of the pin needle in pixels.
        pin_needle_angle (float): Angle of the pin needle in degrees (0 = vertical).
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    width, height = img.size
    min_dim = min(width, height)
    pin_diameter = int(min_dim * (radius_percent / 100))
    pin_radius = pin_diameter // 2

    # --- 1. Create the Pin Face (Circular) ---
    pin_face = Image.new("RGBA", (pin_diameter, pin_diameter), (0, 0, 0, 0))
    draw_face = ImageDraw.Draw(pin_face)
    draw_face.ellipse((0, 0, pin_diameter, pin_diameter), fill=(255, 255, 255, 255))

    # Mask the original image to the circle
    cropped_img_data = img.crop(((width - min_dim) // 2, (height - min_dim) // 2,
                                  (width + min_dim) // 2, (height + min_dim) // 2))
    cropped_img_data = cropped_img_data.resize((pin_diameter, pin_diameter), Image.LANCZOS)
    
    # Use the circle mask for the image
    pin_face_masked = Image.composite(cropped_img_data, pin_face, pin_face)

    # --- 2. Create the Border/Rim (Pseudo-Bevel) ---
    border_diameter = pin_diameter + 2 * border_size
    border_img = Image.new("RGBA", (border_diameter, border_diameter), (0, 0, 0, 0))
    draw_border = ImageDraw.Draw(border_img)
    draw_border.ellipse((0, 0, border_diameter, border_diameter), fill=border_color)

    # Paste the pin face onto the border
    border_img.paste(pin_face_masked, (border_size, border_size), pin_face_masked)

    # --- 3. Create Drop Shadow ---
    # Create an alpha channel from the border_img (where the pin shape is)
    pin_alpha = border_img.split()[3] # Get the alpha channel
    shadow_img = Image.new("RGBA", (border_diameter + shadow_offset_x + shadow_blur_radius * 2,
                                    border_diameter + shadow_offset_y + shadow_blur_radius * 2), (0, 0, 0, 0))
    
    # Draw a black shape for the shadow based on the pin_alpha
    shadow_draw = ImageDraw.Draw(shadow_img)
    
    # Create a mask for the shadow that is the shape of the pin
    # This requires creating a temporary black image for the shape
    temp_black_shape = Image.new("L", (border_diameter, border_diameter), 0)
    temp_draw = ImageDraw.Draw(temp_black_shape)
    temp_draw.ellipse((0, 0, border_diameter, border_diameter), fill=255) # Draw white circle on black
    
    # Use the mask to create the black shadow shape
    shadow_base = Image.composite(Image.new("RGBA", (border_diameter, border_diameter), (0, 0, 0, shadow_opacity)),
                                  Image.new("RGBA", (border_diameter, border_diameter), (0, 0, 0, 0)),
                                  temp_black_shape) # Combine black opaque with transparent using shape mask

    # Paste the black shadow shape into the shadow image
    shadow_img.paste(shadow_base, (shadow_offset_x, shadow_offset_y), temp_black_shape)

    # Apply blur to the shadow
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(shadow_blur_radius))


    # --- 4. Create the Pin Needle ---
    needle_img = Image.new("RGBA", shadow_img.size, (0, 0, 0, 0))
    draw_needle = ImageDraw.Draw(needle_img)

    # Define needle start/end points relative to the center of the pin
    center_x = shadow_img.width // 2
    center_y = shadow_img.height // 2

    # Calculate angle in radians
    angle_rad = math.radians(pin_needle_angle + 90) # +90 because 0 degrees in ImageDraw is along X-axis

    # Needle starts roughly behind the pin's center
    start_x = center_x
    start_y = center_y

    # Calculate end point
    needle_length = pin_diameter * pin_needle_length_ratio
    end_x = start_x + needle_length * math.cos(angle_rad)
    end_y = start_y + needle_length * math.sin(angle_rad)

    # Draw the needle (a line for simplicity, can be more complex for 3D)
    draw_needle.line([(start_x, start_y), (end_x, end_y)], fill=pin_needle_color, width=pin_needle_width)
    
    # Add a simple needle tip
    tip_size = pin_needle_width * 2
    tip_angle_rad = math.radians(pin_needle_angle) # Adjust for tip pointing correctly
    draw_needle.polygon([
        (end_x - tip_size * math.sin(tip_angle_rad), end_y + tip_size * math.cos(tip_angle_rad)),
        (end_x + tip_size * math.sin(tip_angle_rad), end_y - tip_size * math.cos(tip_angle_rad)),
        (end_x + needle_length * 0.05 * math.cos(angle_rad), end_y + needle_length * 0.05 * math.sin(angle_rad)) # A bit further out for point
    ], fill=pin_needle_color)


    # --- 5. Composite all elements ---
    final_image = Image.new("RGBA", shadow_img.size, (0, 0, 0, 0))

    # Paste in order: shadow, needle, pin body
    final_image.paste(shadow_img, (0, 0), shadow_img) # The shadow's alpha makes it opaque
    final_image.paste(needle_img, (0, 0), needle_img)
    
    # Calculate paste position for the pin body to center it relative to the overall image and shadow offsets
    pin_body_offset_x = (final_image.width - border_img.width) // 2
    pin_body_offset_y = (final_image.height - border_img.height) // 2
    
    final_image.paste(border_img, (pin_body_offset_x, pin_body_offset_y), border_img)


    final_image.save(output_path)
    print(f"Pin image saved to {output_path}")


def create_rounded_square_pin_effect(image_path, output_path,
                                     corner_radius_percent=15, border_size=10, border_color=(180, 180, 180, 255),
                                     shadow_offset_x=10, shadow_offset_y=10, shadow_blur_radius=15, shadow_opacity=150,
                                     pin_needle_color=(50, 50, 50, 255), pin_needle_length_ratio=0.8, pin_needle_width=3, pin_needle_angle=20):
    """
    Converts a square image into a rounded square pin-like image with basic 3D effects.
    Parameters are similar to create_circular_pin_effect.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    width, height = img.size
    min_dim = min(width, height)
    pin_side_length = int(min_dim * 0.9) # Adjust this as needed for the overall size of the rounded square
    
    # Calculate corner radius based on pin_side_length
    corner_radius = int(pin_side_length * (corner_radius_percent / 100))
    if corner_radius > pin_side_length / 2: # Prevent radius from being too large
        corner_radius = pin_side_length // 2

    # --- 1. Create the Pin Face (Rounded Square) ---
    pin_face = Image.new("RGBA", (pin_side_length, pin_side_length), (0, 0, 0, 0))
    draw_face = ImageDraw.Draw(pin_face)
    # Draw rounded rectangle
    draw_face.rounded_rectangle((0, 0, pin_side_length, pin_side_length), radius=corner_radius, fill=(255, 255, 255, 255))

    # Mask the original image to the rounded square
    cropped_img_data = img.crop(((width - min_dim) // 2, (height - min_dim) // 2,
                                  (width + min_dim) // 2, (height + min_dim) // 2))
    cropped_img_data = cropped_img_data.resize((pin_side_length, pin_side_length), Image.LANCZOS)
    
    pin_face_masked = Image.composite(cropped_img_data, pin_face, pin_face)

    # --- 2. Create the Border/Rim ---
    border_side_length = pin_side_length + 2 * border_size
    border_img = Image.new("RGBA", (border_side_length, border_side_length), (0, 0, 0, 0))
    draw_border = ImageDraw.Draw(border_img)
    draw_border.rounded_rectangle((0, 0, border_side_length, border_side_length), radius=corner_radius + border_size, fill=border_color)

    border_img.paste(pin_face_masked, (border_size, border_size), pin_face_masked)

    # --- 3. Create Drop Shadow ---
    shadow_img = Image.new("RGBA", (border_side_length + shadow_offset_x + shadow_blur_radius * 2,
                                    border_side_length + shadow_offset_y + shadow_blur_radius * 2), (0, 0, 0, 0))
    
    temp_black_shape = Image.new("L", (border_side_length, border_side_length), 0)
    temp_draw = ImageDraw.Draw(temp_black_shape)
    temp_draw.rounded_rectangle((0, 0, border_side_length, border_side_length), radius=corner_radius + border_size, fill=255)
    
    shadow_base = Image.composite(Image.new("RGBA", (border_side_length, border_side_length), (0, 0, 0, shadow_opacity)),
                                  Image.new("RGBA", (border_side_length, border_side_length), (0, 0, 0, 0)),
                                  temp_black_shape)
    
    shadow_img.paste(shadow_base, (shadow_offset_x, shadow_offset_y), temp_black_shape)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(shadow_blur_radius))

    # --- 4. Create the Pin Needle ---
    needle_img = Image.new("RGBA", shadow_img.size, (0, 0, 0, 0))
    draw_needle = ImageDraw.Draw(needle_img)

    center_x = shadow_img.width // 2
    center_y = shadow_img.height // 2

    angle_rad = math.radians(pin_needle_angle + 90)

    start_x = center_x
    start_y = center_y

    needle_length = pin_side_length * pin_needle_length_ratio
    end_x = start_x + needle_length * math.cos(angle_rad)
    end_y = start_y + needle_length * math.sin(angle_rad)

    draw_needle.line([(start_x, start_y), (end_x, end_y)], fill=pin_needle_color, width=pin_needle_width)
    
    tip_size = pin_needle_width * 2
    tip_angle_rad = math.radians(pin_needle_angle)
    draw_needle.polygon([
        (end_x - tip_size * math.sin(tip_angle_rad), end_y + tip_size * math.cos(tip_angle_rad)),
        (end_x + tip_size * math.sin(tip_angle_rad), end_y - tip_size * math.cos(tip_angle_rad)),
        (end_x + needle_length * 0.05 * math.cos(angle_rad), end_y + needle_length * 0.05 * math.sin(angle_rad))
    ], fill=pin_needle_color)


    # --- 5. Composite all elements ---
    final_image = Image.new("RGBA", shadow_img.size, (0, 0, 0, 0))
    final_image.paste(shadow_img, (0, 0), shadow_img)
    final_image.paste(needle_img, (0, 0), needle_img)
    
    pin_body_offset_x = (final_image.width - border_img.width) // 2
    pin_body_offset_y = (final_image.height - border_img.height) // 2
    
    final_image.paste(border_img, (pin_body_offset_x, pin_body_offset_y), border_img)

    final_image.save(output_path)
    print(f"Pin image saved to {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy square image for testing
    try:
        dummy_img = Image.new("RGBA", (500, 500), (255, 0, 0, 255)) # Red square
        draw_dummy = ImageDraw.Draw(dummy_img)
        draw_dummy.text((50, 200), "Hello Pin!", fill=(255, 255, 255, 255), font_size=80)
        dummy_img.save("dummy_square.png")
        print("Dummy square image created: dummy_square.png")
    except ImportError:
        print("Pillow library not found. Please install it using: pip install Pillow")
        exit()


    # Example 1: Circular Pin
    print("\nCreating circular pin...")
    create_circular_pin_effect(
        #image_path="dummy_square.png",
        image_path="logo_dlmuse_init.png",
        output_path="output_pin_circular.png",
        radius_percent=80, # Make the circle a bit smaller than the image
        border_size=15,
        border_color=(200, 200, 200, 255), # Lighter grey
        shadow_offset_x=15,
        shadow_offset_y=15,
        shadow_blur_radius=20,
        shadow_opacity=180,
        pin_needle_length_ratio=0.7,
        pin_needle_angle=30 # More diagonal
    )

    # Example 2: Rounded Square Pin
    print("\nCreating rounded square pin...")
    create_rounded_square_pin_effect(
        image_path="dummy_square.png",
        output_path="output_pin_rounded_square.png",
        corner_radius_percent=20, # Higher radius for more rounded corners
        border_size=12,
        border_color=(150, 150, 150, 255), # Darker grey
        shadow_offset_x=8,
        shadow_offset_y=8,
        shadow_blur_radius=12,
        shadow_opacity=120,
        pin_needle_width=4,
        pin_needle_angle=10 # Less diagonal
    )

    # Example 3: Using a real image (replace with your image path)
    # Ensure 'your_image.png' is a square image for best results.
    # print("\nCreating circular pin from another image...")
    # create_circular_pin_effect(
    #     image_path="path/to/your/square_image.png",
    #     output_path="output_pin_from_your_image.png",
    #     radius_percent=90,
    #     border_size=10,
    #     border_color=(190, 190, 190, 255),
    #     shadow_offset_x=12,
    #     shadow_offset_y=12,
    #     shadow_blur_radius=18,
    #     shadow_opacity=160
    # )
