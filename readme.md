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

3. **Run the backend or test script:**
   ```sh
   python3 test.py
   ```

# The crux of it:

- train.py was used to train the model 

- So the kaggle dataset csv file had the correct angles for each workout. 
Here is that dataset: https://www.kaggle.com/datasets/mohamedkhapiry/rehab24-6

- Used that csv, put it into a RandomForestClassifier ml model and trained it, which takes in angles and tells if workout is correct or not

- Yolo detects the angles and sends it to that model (simple)


## Right now what's happening:

- Frontend sends the frames as base64 encoded
- Recieves it in ``test.py`` through the ws endpoint
- Converts them into proper frames, gives them to the yolo model, and it tells if its correct or not
- The angles from each rep is sent to the ``llm_feedback.py`` which gives more feedback
- Then this data is recieved in the frontend and displayed.



