"""Create a placeholder board image for Snakes and Ladders.
Run this script to generate the board image: python create_board.py"""
from pathlib import Path
from PIL import Image, ImageDraw

# Board dimensions: 10x10 grid, 60px tiles, 50px margins
tile_width = 60
tile_height = 60
rows = 10
cols = 10
start_x = 50
start_y = 50
board_width = start_x * 2 + (cols * tile_width)
board_height = start_y * 2 + (rows * tile_height)

# Create image
img = Image.new("RGB", (board_width, board_height), color=(240, 240, 240))
draw = ImageDraw.Draw(img)

# Draw grid
for row in range(rows):
    for col in range(cols):
        x = start_x + (col * tile_width)
        y = start_y + (row * tile_height)
        
        # Alternate tile colors
        if (row + col) % 2 == 0:
            tile_color = (255, 255, 255)  # White
        else:
            tile_color = (200, 200, 200)  # Light gray
        
        draw.rectangle(
            [x, y, x + tile_width - 1, y + tile_height - 1],
            fill=tile_color,
            outline=(100, 100, 100),
            width=1
        )
        
        # Calculate tile number (1-100, starting from bottom-left A1)
        # A1 = tile 1, J10 = tile 100
        tile_num = (rows - row - 1) * cols + col + 1
        
        # Draw tile number
        text = str(tile_num)
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (tile_width - text_width) // 2
        text_y = y + (tile_height - text_height) // 2
        
        draw.text((text_x, text_y), text, fill=(0, 0, 0))

# Save image
output_path = Path(__file__).parent / "snakes_ladders_board.png"
img.save(output_path)
print(f"Created board image: {output_path}")

