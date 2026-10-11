FROM python:3.10-slim

# התקנת כלי מערכת הכרחיים (FFmpeg, git, python3-pip)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# התקנת תלויות פייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת הקוד
COPY . .

# יצירת תיקיית עבודה זמנית לקבצים
RUN mkdir -p /tmp/jobs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]