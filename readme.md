## How to Run This Project

```sh
### Make sure to add the OPENAI_API_KEY in the .env file 
```

1. **Start the frontend:**
   ```sh
   cd frontend
   npm install
   npm run dev
   ```

2. **Install Python dependencies:**
   ```sh
   pip3 install -r requirements.txt
   ```

3. **Run the backend server:**
   ```sh
   python3 server.py
   ```
   The server will start on `http://0.0.0.0:8000`

## Project Structure

The codebase is modularized into the following components:

- **`server.py`** - FastAPI application with WebSocket endpoints for real-time frame processing
- **`detect.py`** - Model loading (YOLO pose estimation + RandomForest classifier) and frame processing logic
- **`utils/utils.py`** - Keypoint extraction and angle calculation utilities
- **`utils/rep_tracking.py`** - Rep counting logic, state management, and exercise tracking
- **`utils/analysis.py`** - Statistics calculation, grading, and error analysis
- **`utils/feedback.py`** - LLM feedback integration and session summary generation
- **`utils/serialization.py`** - Data conversion utilities for JSON serialization
- **`llm_feedback.py`** - OpenAI integration for generating personalized exercise feedback

## How It Works

### Training Phase:
- `train.py` was used to train the RandomForest classifier model
- The Kaggle dataset CSV file had the correct angles for each workout
- Dataset: https://www.kaggle.com/datasets/mohamedkhapiry/rehab24-6
- The CSV data was used to train a RandomForestClassifier that takes joint angles as input and predicts if the workout form is correct or not

### Runtime Flow:

1. **Frontend** sends video frames as base64 encoded images via WebSocket
2. **`server.py`** receives frames through the `/ws` WebSocket endpoint
3. **`detect.py`** processes each frame:
   - Converts base64 to numpy array
   - Runs YOLO pose estimation to detect keypoints
   - Extracts joint angles from keypoints
   - Feeds angles to the trained RandomForest classifier to determine if form is correct
4. **`utils/rep_tracking.py`** tracks rep counting:
   - Monitors arm movement (up/down states)
   - Counts reps per arm (5 reps right, then 5 reps left)
   - Calculates rep quality metrics
5. **`utils/feedback.py`** triggers LLM feedback:
   - Every 2 reps, sends exercise data to `llm_feedback.py`
   - Generates personalized feedback using OpenAI API
   - Provides actionable tips and recommendations
6. **Frontend** receives and displays:
   - Real-time form correctness indicators
   - Rep counts and grades
   - LLM-generated feedback
   - Session summaries



