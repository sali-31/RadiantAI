from PIL import Image;
import io

def scrub_image_metadata(image_bytes: bytes) -> bytes:
    """
    This function removes EXIF metadata from an image to protect user privacy.

    Strategy:
    1. Open the image from the raw bytes.
    2. Create a new image object with the same mode (RGB/RGBA) and size.
    3. Copy the pixel data from the original to the new image.
    4. This process drops all non-pixel data (metadata, EXIF, color profiles).
    """
    try:
        # 1. Open the image from memory
        img = Image.open(io.BytesIO(image_bytes))

        # 2. Create a new blank image with the same settings
        image_no_exif = Image.new(img.mode, img.size)

        # 3. Copy the pixel data
        # fyi: .getdata() returns the sequence of pixel values
        image_no_exif.putdata(img.getdata())

        # 4. Save the new image to a memory buffer
        output_buffer = io.BytesIO()

        # We will use PNG because it is lossless (doesn't degrade quality like JPEG would)
        # and doesn't automatically add metadata
        image_no_exif.save(output_buffer, format="PNG")

        # Now we'll return the raw bytes of the new, clean image
        return output_buffer.getvalue()
    
    except Exception as e:
        # Reject images that can't be scrubbed, or cover other errors
        print(f"Error scrubbing metadata: {e}")
        raise e
