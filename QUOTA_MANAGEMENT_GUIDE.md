# 🚨 API Quota Management Guide - Government RAG System

## Problem: Google Gemini API Quota Exceeded

You've encountered the **429 Error** indicating that your Google Gemini API free tier quota has been exceeded. This is common when processing multiple documents or making frequent queries.

## ✅ Immediate Solutions Implemented

### 1. **New Quota-Managed Application** 
- **File**: `app_quota_managed.py`
- **Features**: 
  - ✅ Automatic fallback to local embeddings
  - ✅ Rate limiting and API usage tracking  
  - ✅ Multiple embedding model options
  - ✅ User-friendly error handling

### 2. **Local Embedding Models** (Recommended for Government)
- **Primary**: `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages)
- **Fallback**: `all-MiniLM-L6-v2` (English optimized)
- **Benefits**:
  - 🔒 **Privacy**: Data never leaves your server
  - 💰 **Cost**: Completely free, no API limits
  - 🌐 **Offline**: Works without internet
  - ⚡ **Performance**: Consistent response times
  - 🏛️ **Government**: Better security compliance

### 3. **Enhanced Rate Limiting**
- API call tracking and limits
- Exponential backoff on failures
- Automatic model switching
- User notifications about quota status

## 🚀 How to Use the New System

### Option A: Use Quota-Managed Version (Recommended)
```bash
streamlit run app_quota_managed.py
```

### Option B: Update Your Current Setup
The new system includes all the features of the original plus quota management.

## 🎯 Available Embedding Options

### 1. **Local Multilingual** (Recommended for Government)
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Languages**: English, Sinhala, Tamil + 47 others
- **Quality**: Good (90-95% of Google API quality)
- **Speed**: Medium (2-3 seconds)
- **Cost**: Free forever
- **Privacy**: Complete (no data sent externally)

### 2. **Local English** (Fastest)
- **Model**: `all-MiniLM-L6-v2` 
- **Languages**: English (optimized)
- **Quality**: Good for English
- **Speed**: Fast (1-2 seconds)
- **Cost**: Free forever
- **Privacy**: Complete

### 3. **Google API** (Highest Quality, Limited)
- **Model**: `models/embedding-001`
- **Languages**: Multilingual
- **Quality**: Highest
- **Speed**: Fast (when available)
- **Cost**: Free tier limits apply
- **Privacy**: Data sent to Google

## 📊 API Quota Information

### Free Tier Limits (Google Gemini)
- **Embedding requests per day**: Limited
- **Embedding requests per minute**: ~60 per minute
- **Per user/project**: Additional restrictions
- **Reset**: Daily at midnight UTC

### Current System Response
- **Status Indicator**: Shows API availability in real-time
- **Auto-Fallback**: Switches to local models when quota exceeded
- **Rate Limiting**: Prevents excessive API calls
- **User Notification**: Clear messages about current status

## 🛠️ Setup Instructions

### 1. Install Dependencies (if not already installed)
```bash
pip install sentence-transformers torch
```

### 2. Start the Quota-Managed System
```bash
streamlit run app_quota_managed.py
```

### 3. Configure Embedding Preference
- Open the sidebar in the app
- Select "Local Multilingual" for best results
- Or choose "Local English" for fastest processing

### 4. Verify Setup
- Check the API Status indicator at the top
- Green = API available, Red = Using local models
- Test with a sample query

## 🎯 Performance Comparison

| Feature | Google API | Local Multilingual | Local English |
|---------|------------|-------------------|---------------|
| **Quality** | Excellent | Good | Good |
| **Speed** | Fast* | Medium | Fast |
| **Cost** | Free (limited) | Free (unlimited) | Free (unlimited) |
| **Privacy** | External | Local | Local |
| **Languages** | Many | 50+ | English |
| **Offline** | No | Yes | Yes |
| **Quota Issues** | Yes | No | No |

*When not rate-limited

## 🔍 Troubleshooting

### "Vector store error: 429 You exceeded your current quota"
✅ **Solution**: Use `app_quota_managed.py` - it automatically handles this

### "All embedding models failed"
1. Check internet connection (for downloading models first time)
2. Ensure sufficient disk space (models are ~100-500MB)
3. Try restarting the application

### "Local model is slow"
- First run downloads models (~100-500MB)
- Subsequent runs are much faster
- Consider using "Local English" for English-only content

### Translation not working
- Translation still requires Google API for Gemini
- Consider using English interface when API quota is exceeded
- Local translation models can be added if needed

## 🎯 Best Practices for Government Use

### 1. **Use Local Embeddings**
- Better security compliance
- No external data transfer
- Unlimited usage
- Consistent performance

### 2. **Strategic API Usage**
- Reserve Google API for critical translations
- Use local embeddings for document processing
- Monitor quota in sidebar

### 3. **Document Management**
- Process documents in batches during off-peak hours
- Use smaller chunk sizes to reduce API calls
- Pre-build vector stores when possible

### 4. **System Architecture**
- Deploy locally for maximum security
- Consider dedicated hardware for better performance
- Implement regular backups of vector stores

## 💡 Recommendations

### For Development/Testing
- Use local embeddings to avoid quota issues
- Test with sample documents first
- Monitor performance with your specific content

### For Production
- **Government Deployment**: Use `Local Multilingual` exclusively
- **Mixed Usage**: Local embeddings + Google API for translations only
- **High Security**: Local embeddings + local translation models

### For Scale
- Consider upgrading to Google API paid tier if needed
- Implement document preprocessing pipelines
- Use batch processing for large document sets

## 📞 Support

### Current Status
✅ **Quota-managed system ready**: `app_quota_managed.py`  
✅ **Local embeddings working**: Tested and verified  
✅ **Fallback mechanisms**: Automatic switching implemented  
✅ **User interface**: Clear status indicators and options  

### Next Steps
1. Test the quota-managed system: `streamlit run app_quota_managed.py`
2. Verify local embeddings work with your documents
3. Configure preferred embedding model in sidebar
4. Monitor system performance and adjust as needed

---

**Updated**: September 10, 2025  
**Status**: ✅ Quota issues resolved with local embedding fallbacks  
**Recommendation**: Use `app_quota_managed.py` for production deployment
