# Set up a MoDaCor data processing server container
# Build:
#   podman build . -t modacor
# Run, for debugging, seeing the log output:
#   podman run -it --rm -p 8700:8700 modacor

# Use official lightweight Python image
FROM python:3.14-slim

ARG TZ
# Set the environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
# Set the timezone by given argument
ENV TZ=${TZ}

# Install build deps, then runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# create the data mount point
WORKDIR /mnt/vsi-db
# get the MoDaCor code
WORKDIR /app
COPY . /app

# Install pip and project dependencies (pip here for simplicity)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir . .[server]

# Expose port
EXPOSE 8700

# Run the app with Uvicorn
CMD ["uvicorn", "modacor.server.serve:main", "--factory", "--host", "0.0.0.0", "--port", "8700"]
