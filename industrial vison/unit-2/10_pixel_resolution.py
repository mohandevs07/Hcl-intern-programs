import cv2

image = cv2.imread("../images/industrial.jpg")

reference_width_mm = 100
reference_width_pixels = 500

pixel_resolution = (
    reference_width_mm / reference_width_pixels
)

print("Reference Object Width:", reference_width_mm, "mm")
print("Reference Object in Image:", reference_width_pixels, "pixels")
print(f"Pixel Resolution: {pixel_resolution:.3f} mm/pixel")

object_width_pixels = 750

estimated_width_mm = (
    object_width_pixels * pixel_resolution
)

print(
    f"Estimated Object Width: "
    f"{estimated_width_mm:.2f} mm"
)