# Use Python 3.8 as the base image
FROM python:3.8

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file to the working directory
RUN cd / && \
    git clone https://github.com/CBICA/spare_score && \
    cd /spare_score && pip install . 

WORKDIR /spare_score

# Set the command to run the Python script
CMD ["python", "merge_ROI_demo_and_test.py"]
