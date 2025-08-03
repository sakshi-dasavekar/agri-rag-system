# 🌾 Agricultural Expert System - Deployment Guide

This guide will help you deploy your Agricultural RAG System to various platforms.

## 📋 Prerequisites

1. **Groq API Key**: Get your API key from [Groq Console](https://console.groq.com/)
2. **Processed Data**: Ensure your `rag_storage_filtered/` folder contains the processed datasets
3. **Python 3.8+**: Required for all dependencies

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add agricultural RAG system"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set environment variables:
     - `GROQ_API_KEY`: Your Groq API key
   - Deploy!

3. **Required Files**:
   - `streamlit_app.py` (main app)
   - `requirements.txt` (dependencies)
   - `rag_storage_filtered/` (processed data)

### Option 2: Heroku

1. **Create Procfile**:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Set Environment Variables**:
   ```bash
   heroku config:set GROQ_API_KEY=your_api_key_here
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Railway

1. **Connect Repository**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository

2. **Set Environment Variables**:
   - `GROQ_API_KEY`: Your Groq API key

3. **Deploy**:
   - Railway will automatically detect and deploy your Streamlit app

### Option 4: Local Deployment

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variable**:
   ```bash
   # Windows
   set GROQ_API_KEY=your_api_key_here
   
   # Linux/Mac
   export GROQ_API_KEY=your_api_key_here
   ```

3. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Project Structure

```
your-project/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md            # This guide
├── .env                     # Environment variables (local only)
├── datasets/                # Original CSV datasets
├── rag_storage_filtered/    # Processed embeddings and chunks
│   ├── Bio-Pesticides and Bio-Fertilizers/
│   ├── Crop Insurance/
│   ├── Cultivation Conditions/
│   └── ... (other datasets)
└── README.md               # Project documentation
```

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key for LLM access | ✅ Yes |

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'langchain_groq'"**:
   ```bash
   pip install -r requirements.txt
   ```

2. **"Groq API key not found"**:
   - Set the `GROQ_API_KEY` environment variable
   - For local development, create a `.env` file

3. **"No dataset folders found"**:
   - Ensure `rag_storage_filtered/` folder exists
   - Run the data processing script first

4. **Memory Issues**:
   - Reduce the number of datasets processed
   - Use smaller chunk sizes in processing

## 📊 Performance Optimization

1. **Reduce Memory Usage**:
   - Process fewer datasets
   - Use smaller embedding models
   - Implement caching

2. **Improve Speed**:
   - Use GPU acceleration if available
   - Optimize chunk sizes
   - Implement request batching

## 🔒 Security Considerations

1. **API Key Security**:
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Data Privacy**:
   - Ensure agricultural data is properly anonymized
   - Follow data protection regulations

## 📈 Monitoring

1. **Streamlit Cloud**: Built-in analytics
2. **Custom Monitoring**: Add logging to track usage
3. **Error Tracking**: Monitor for API failures

## 🆘 Support

If you encounter issues:

1. Check the error messages in the Streamlit app
2. Verify all dependencies are installed
3. Ensure environment variables are set correctly
4. Check that processed data exists

## 🎯 Next Steps

After deployment:

1. **Test the Application**: Try sample questions
2. **Monitor Performance**: Check response times
3. **Gather Feedback**: Collect user feedback
4. **Iterate**: Improve based on usage patterns

---

**Happy Deploying! 🌾** 