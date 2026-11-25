FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MRI_HOST=0.0.0.0 GRADIO_SERVER_PORT=7860 MRI_CLASSIC=0
WORKDIR /app
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
RUN useradd -m -u 1000 user
COPY --chown=user:user app.py mri_model.py web_server.py ./
COPY --chown=user:user web ./web
COPY --chown=user:user source/Image_Detection_Model.h5 ./source/Image_Detection_Model.h5
USER user
EXPOSE 7860
CMD ["python", "app.py"]

