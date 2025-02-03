# Dockerfile
# Use the official Pathway base image
FROM pathwaycom/pathway:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Tesseract OCR and the English language data file
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng

# Set the TESSDATA_PREFIX environment variable
ENV TESSDATA_PREFIX=/usr/share/tesseract/tessdata

# Copy the rest of the application code to the container
COPY . .

# Create the uploads directory
RUN mkdir -p /app/data

# Expose the FastAPI port (9000)
EXPOSE 9000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]