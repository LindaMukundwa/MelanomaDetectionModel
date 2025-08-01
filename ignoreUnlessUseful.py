import streamlit as st

# this file is to check the tensor flow versions code if necessary
# testing what version of tensorflow i have 
st.write("## TensorFlow Version Compatibility Check")

# Check versions
st.write(f"**Current TensorFlow version:** {tf.__version__}")
try:
    keras_version = tf.keras.__version__
    st.write(f"**Current Keras version:** {keras_version}")
except AttributeError:
    st.write("**Current Keras version:** Integrated with TensorFlow (no separate version)")

# Check if this matches what you used in Colab
st.write("**Expected Colab versions (as of 2024):**")
st.write("- TensorFlow: 2.15.x or 2.16.x")
st.write("- If your Colab used a different version, this explains the error")

# Check model files
st.write("## Model Files Status")
model_files = [
    'models/cnn_model.keras',
    'models/abcd_model.keras', 
    'models/combined_model.keras',
    'models/abcd_scaler.pkl'
]

for file in model_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        st.write(f"✅ {file} ({size:,} bytes)")
    else:
        st.write(f"❌ {file} missing")

# Try to identify the exact error
st.write("## Attempting Model Load with Detailed Error Info")
try:
    # Try loading with standard method
    model = tf.keras.models.load_model('models/cnn_model.keras', compile=False)
    st.success("✅ Model loaded successfully!")
    st.write(f"Model input shape: {model.input_shape}")
    st.write(f"Model output shape: {model.output_shape}")
except Exception as e:
    st.error(f"❌ Detailed error: {str(e)}")
    st.write("This confirms the version compatibility issue.")
    st.write("**Solution**: The models need to be re-saved with the current TensorFlow version.")