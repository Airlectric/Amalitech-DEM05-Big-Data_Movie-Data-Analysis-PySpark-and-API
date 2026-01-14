FROM apache/spark-py:v3.5.1

USER root

# Install extra system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install --no-cache-dir \
    jupyterlab \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    sqlalchemy \
    psycopg2-binary \
    findspark

# Create working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
