import streamlit as st
import subprocess
import shutil
import os
import glob

output_directory = 'Data/Processed/online'
figures_directory = 'Figure-Prediction'

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def update_chatbox(new_message):
    st.session_state.chat_history.append(new_message)

with st.form("my_form"):
    st.write("Epidemics Forecasting with GradMetaPopulation")
    city = st.selectbox('Pick a city', ['bogota'])
    device = st.selectbox('Pick a device', ['cpu', 'gpu'])

    csv_file_1 = st.file_uploader("Upload the Train Dataset", type=["csv"])
    csv_file_2 = st.file_uploader("Upload the Test Dataset", type=["csv"])
    pt_file = st.file_uploader("Upload the Private Transaction Dataset", type=["pt"])

    status = st.form_submit_button('Start Running')

if status:
    if csv_file_1 and csv_file_2 and pt_file:
        # Save the first CSV file
        with open(os.path.join(output_directory, csv_file_1.name), "wb") as f:
            shutil.copyfileobj(csv_file_1, f)
        
        # Save the second CSV file
        with open(os.path.join(output_directory, csv_file_2.name), "wb") as f:
            shutil.copyfileobj(csv_file_2, f)
        
        # Save the .pt file
        with open(os.path.join(output_directory, pt_file.name), "wb") as f:
            shutil.copyfileobj(pt_file, f)
        
        st.text("Files have been saved!")
    else:
        st.warning("Please upload all three files before saving.")

    process = subprocess.Popen(
        ["python", '-u', "src/main.py", "--state", "MA", "--disease", city, "--joint", "--dev", "0" if device == "gpu" else device, "--model_name", "meta", "--disease", "bogota", "-date", "0_moving", "-week", 49],
        stdout=subprocess.PIPE,
        universal_newlines=True
    )

    output_box = st.empty()
    max_lines = 10  # Display only the last 20 lines

    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            update_chatbox(line.strip())
            # Display only the last 'max_lines' lines in the text area
            output_box.text_area(
                "Output",
                "\n".join(st.session_state.chat_history[-max_lines:]),
                height=300
            )
    
    # Display the generated image after the subprocess completes
    process.wait()  # Ensure the process has finished
    image_files = glob.glob(os.path.join(figures_directory, "*.png"))  # Adjust extension if needed (e.g., .jpg)

    if image_files:
        latest_image = max(image_files, key=os.path.getctime)  # Get the most recent image
        st.image(latest_image, caption="Generated Prediction Image", use_column_width=True)
    else:
        st.warning("No image found in the Figures-Prediction directory.")
