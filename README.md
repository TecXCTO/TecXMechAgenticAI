# TecXMechAgenticAI


# for Google Gemini llm
# Configuration

# Set up Google Cloud credentials.

# You may set the following environment variables in your shell, or in a .env file instead.

export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<your-project-id>
export GOOGLE_CLOUD_LOCATION=<your-project-location>
export GOOGLE_CLOUD_STORAGE_BUCKET=<your-storage-bucket>  # Only required for deployment on Agent Engine

# Authenticate your GCloud account.

gcloud auth application-default login
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT

# If you'd prefer to run the agent locally without using Google Vertex AI or Cloud dependencies, you may set the following environment variables in your shell, or in a .env file instead.

export GOOGLE_GENAI_USE_VERTEXAI=false
export GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
