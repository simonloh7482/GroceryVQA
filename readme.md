# Grocery VQA
1. Read the readme file from dataset to prepare the dataset.
2. Install the required packages by running 'pip install -r requirements.txt'
3. Run scripts 2 & 3 to preprocess the dataset, and 4 to train the model.
4. Run script 5 to evaluate the model. (Optional)
5. Run script 6 to host a VQA server.
6. Run script 8 to test the VQA server in case the mobile application is not available. Modify the script to target the server's IP address before running.

## Important files:
1. config/default.yaml: Most of the configurations can be modified here.

2. preprocessing/CreateVocab.py: To create vocabs based on train annotations in vqa_data/Annotations

3. preprocessing/ImageFeatureExtractor.py: To extract image features in vqa_data/Images using pretrained ResNet152.

4. train.py: train the model using the extracted features and vocabs in vqa_data folder. Model is saved in logs/vizwiz folder

5. predict.py: Test the VQA model stored in logs/vizwiz folder and store the predictions in /predictions folder

6. server.py: script to run a HTTP server to accept VQA requests and return the answer to the clients.

7. vqa_predict.py: a modified predict function to work with server.py

8. desktopAppGUI.py: The client application for testing out the VQA Model and the server.