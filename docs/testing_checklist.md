# 🧪 RadiantAI Testing Checklist

## Quick Start

### Option 1: One-Command Startup (Recommended)
```bash
chmod +x start-dev.sh
./start-dev.sh
```

### Option 2: Manual Startup

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## 🔍 Pre-Flight Checks

### Environment Setup
- [ ] `.env` file exists in project root
- [ ] `.env` contains `GOOGLE_API_KEY`
- [ ] `.env` contains `AWS_ACCESS_KEY_ID`
- [ ] `.env` contains `AWS_SECRET_ACCESS_KEY`
- [ ] `.env` contains `AWS_REGION`
- [ ] `.env` contains `S3_BUCKET_NAME`

### Dependencies Installed
- [ ] Backend: `pip install -r backend/requirements.txt`
- [ ] Frontend: `cd frontend && npm install`

### Servers Running
- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:5173
- [ ] API docs accessible at http://localhost:8000/docs

---

## 📸 Camera Capture Feature Testing

### 1. Camera Access & Permissions

**Test Case 1.1: Request Camera Permission**
- [ ] Navigate to http://localhost:5173
- [ ] Click "Use Camera to Capture Photo" button
- [ ] Browser prompts for camera permission
- [ ] Click "Allow"
- [ ] **Expected:** Video preview appears with mirrored view (selfie mode)

**Test Case 1.2: Camera Permission Denied**
- [ ] Clear site permissions in browser settings
- [ ] Click "Use Camera to Capture Photo" button
- [ ] Click "Block" on permission prompt
- [ ] **Expected:** Red error banner appears: "Camera permission denied. Please allow camera access in your browser settings."
- [ ] Click "Dismiss" button
- [ ] **Expected:** Error banner disappears

**Test Case 1.3: No Camera Available**
- [ ] Test on device without camera (if available)
- [ ] **Expected:** Error message: "No camera found on this device."

### 2. Live Video Preview

**Test Case 2.1: Video Mirroring**
- [ ] Start camera
- [ ] Move hand in front of camera
- [ ] **Expected:** Video shows mirrored view (like looking in a mirror)
- [ ] Verify video is smooth (no lag/stuttering)

**Test Case 2.2: Video Quality**
- [ ] Check video resolution
- [ ] **Expected:** Clear, high-quality preview (1280x720 ideal)
- [ ] Verify proper lighting visibility

**Test Case 2.3: Cancel Camera**
- [ ] Click "Cancel" button while camera is active
- [ ] **Expected:**
  - Video preview disappears
  - Camera light turns off
  - Returns to file upload mode

### 3. Photo Capture

**Test Case 3.1: Capture Photo**
- [ ] Start camera
- [ ] Position face/skin area in frame
- [ ] Click "Capture Photo" button
- [ ] **Expected:**
  - Photo captured instantly
  - Preview shows captured image
  - Camera stops (light turns off)
  - "Retake" and "Upload & Analyze" buttons appear

**Test Case 3.2: Captured Image Quality**
- [ ] Verify captured image is clear (not blurry)
- [ ] Verify captured image is NOT mirrored (correct orientation for analysis)
- [ ] Verify image aspect ratio preserved

**Test Case 3.3: Retake Photo**
- [ ] Capture a photo
- [ ] Click "Retake" button
- [ ] **Expected:**
  - Returns to camera mode
  - Camera restarts
  - Previous capture discarded

**Test Case 3.4: Multiple Retakes**
- [ ] Capture photo
- [ ] Retake 3 times
- [ ] **Expected:** Each retake works smoothly, no memory leaks

### 4. Upload & Analysis

**Test Case 4.1: Upload Captured Photo**
- [ ] Capture photo
- [ ] Click "Upload & Analyze" button
- [ ] **Expected:**
  - Upload status shows "Uploading..."
  - After ~2-5 seconds: "Upload was successful! ✅"
  - Backend logs show image received

**Test Case 4.2: Verify Backend Processing**
- [ ] Check backend logs for:
  - [ ] "Receiving file skin-analysis-[timestamp].jpg for user [user_id]"
  - [ ] "Scrubbing image metadata..."
  - [ ] "Sending image to AI Ensemble..."
  - [ ] "AI analysis completed in X.XXs"
  - [ ] "✓ Upload successful: [S3 URL]"

**Test Case 4.3: File Format**
- [ ] Captured file should be JPEG
- [ ] File size should be reasonable (< 2MB typically)
- [ ] MIME type: `image/jpeg`

### 5. File Upload (Original Feature)

**Test Case 5.1: File Upload Still Works**
- [ ] Refresh page
- [ ] Click "Click to upload" area
- [ ] Select image file from computer
- [ ] Click "Upload for Analysis"
- [ ] **Expected:** Upload succeeds as before

**Test Case 5.2: Both Upload Methods Work**
- [ ] Upload file → Success
- [ ] Refresh page
- [ ] Use camera capture → Success
- [ ] **Expected:** Both methods work independently

---

## 🐛 Edge Cases & Error Handling

### Memory Management

**Test Case 6.1: No Memory Leaks**
- [ ] Open browser DevTools → Memory tab
- [ ] Take heap snapshot (Snapshot 1)
- [ ] Capture photo 10 times (with retakes)
- [ ] Take heap snapshot (Snapshot 2)
- [ ] **Expected:** Memory difference < 10MB (blob URLs cleaned up)

**Test Case 6.2: Blob URL Cleanup**
- [ ] Open DevTools → Console
- [ ] Capture photo
- [ ] Check for `blob:http://localhost:5173/...` URLs
- [ ] Retake photo
- [ ] **Expected:** Previous blob URL revoked (no errors in console)

### State Management

**Test Case 7.1: Mode Transitions**
- [ ] Start in `file` mode → Click camera → `camera` mode
- [ ] Capture → `preview` mode
- [ ] Upload → `file` mode
- [ ] **Expected:** Each transition works smoothly

**Test Case 7.2: Cancel from Camera**
- [ ] Start camera
- [ ] Cancel
- [ ] **Expected:** Returns to file mode, camera stopped

**Test Case 7.3: UI Consistency**
- [ ] In `file` mode: Only file upload + camera button visible
- [ ] In `camera` mode: Only video + capture/cancel buttons visible
- [ ] In `preview` mode: Only captured image + retake/upload buttons visible

### Browser Compatibility

**Test Case 8.1: Chrome/Edge (Desktop)**
- [ ] Test all camera features
- [ ] **Expected:** Full functionality

**Test Case 8.2: Firefox (Desktop)**
- [ ] Test all camera features
- [ ] **Expected:** Full functionality

**Test Case 8.3: Safari (Desktop)**
- [ ] Test all camera features
- [ ] **Expected:** Full functionality

**Test Case 8.4: Mobile Safari (iOS)**
- [ ] Test on iPhone
- [ ] Verify `playsInline` prevents fullscreen takeover
- [ ] **Expected:** Video stays inline, doesn't go fullscreen

**Test Case 8.5: Chrome Mobile (Android)**
- [ ] Test on Android device
- [ ] Test front camera
- [ ] **Expected:** Full functionality

---

## 🔧 Backend Integration Testing

### Test Case 9.1: End-to-End Flow
- [ ] Capture photo of skin with visible blemishes
- [ ] Upload & Analyze
- [ ] **Expected Backend Response:**
  ```json
  {
    "message": "Upload and analysis successful",
    "s3_path": "https://...",
    "filename": "skin-analysis-1234567890.jpg",
    "ai_analysis": {
      "analysis": "{\"characterization\": \"...\", ...}",
      "vision_objects": [...]
    },
    "product_recommendations": {
      "bundle": [...],
      "total_cost": 50.0
    }
  }
  ```

### Test Case 9.2: Image Metadata Stripped
- [ ] Capture photo with camera (includes timestamp)
- [ ] Upload to backend
- [ ] Download from S3
- [ ] Check EXIF data: `exiftool [downloaded-image].jpg`
- [ ] **Expected:** All metadata removed (privacy feature)

### Test Case 9.3: Concurrent Uploads
- [ ] Open two browser tabs
- [ ] Tab 1: Capture and upload
- [ ] Tab 2: Capture and upload simultaneously
- [ ] **Expected:** Both succeed independently

---

## 📊 Performance Testing

### Test Case 10.1: Capture Speed
- [ ] Measure time from clicking "Capture Photo" to preview appearing
- [ ] **Expected:** < 500ms

### Test Case 10.2: Upload Speed
- [ ] Measure time from "Upload & Analyze" to success message
- [ ] **Expected:** 2-5 seconds (depends on network + AI analysis)

### Test Case 10.3: Camera Startup Time
- [ ] Measure time from clicking "Use Camera" to video preview
- [ ] **Expected:** 1-3 seconds (depends on hardware)

---

## ✅ Final Verification

### Code Quality
- [ ] No TypeScript errors in frontend
- [ ] No Python errors in backend
- [ ] No console errors in browser DevTools
- [ ] No warnings about unrevoked blob URLs

### User Experience
- [ ] All buttons have appropriate labels
- [ ] Loading states clear ("Starting Camera...", "Uploading...")
- [ ] Error messages user-friendly
- [ ] Mobile-responsive design

### Documentation
- [ ] Code comments explain camera logic
- [ ] README updated with camera feature
- [ ] This testing checklist complete

---

## 🚨 Known Issues / Limitations

- Camera capture requires HTTPS in production (localhost works with HTTP)
- iOS Safari may require user to manually enable camera in settings
- Some browsers may not support `facingMode: 'user'` constraint
- Blob URLs must be revoked to prevent memory leaks (implemented)

---

## 🎉 Success Criteria

All tests pass when:
- ✅ Camera capture works on desktop browsers
- ✅ Mobile devices can capture photos
- ✅ No memory leaks after multiple captures
- ✅ Backend receives and processes captured images
- ✅ File upload method still works
- ✅ No TypeScript or runtime errors
- ✅ Professional UX (loading states, error handling)

---

## 🔗 Useful Commands

**View backend logs:**
```bash
tail -f logs/backend.log
```

**View frontend logs:**
```bash
tail -f logs/frontend.log
```

**Check processes:**
```bash
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
```

**Stop all servers:**
```bash
pkill -f "uvicorn src.main:app"
pkill -f "vite"
```

**Clear browser cache:**
- Chrome: DevTools → Application → Clear Storage
- Firefox: DevTools → Storage → Clear All
- Safari: Develop → Empty Caches
