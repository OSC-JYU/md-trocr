from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np

# Load the model and processor
processor = TrOCRProcessor.from_pretrained("./models/multicentury-htr-model/processor")
model = VisionEncoderDecoderModel.from_pretrained("./models/multicentury-htr-model")

# Open an image of handwritten text
image = Image.open("test/mask_000.png").convert("RGB")
# image = Image.open("htr2.jpg")
# image = np.array(image)
# image = image[:, :, :3]

# Preprocess and predict
# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

# print(generated_text)


# Preprocess and predict
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# Generate text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)