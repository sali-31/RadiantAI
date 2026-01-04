import React, { useEffect, useState } from "react"
import type { AnalysisResponse } from "../types";
import { removeBackground } from "@imgly/background-removal";

type CaptureMode = 'file' | 'camera' | 'preview';

interface ImageUploadProps {
    userId: string;
    onAnalysisComplete: (data: AnalysisResponse) => void;
}

export const ImageUpload = ({ userId, onAnalysisComplete }: ImageUploadProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploadStatus, setUploadStatus] = useState<string>('');
    const [captureMode, setCaptureMode] = useState<CaptureMode>('file'); // Controls the displayed UI (file picker/ live camera / preview)
    const [stream, setStream] = useState<MediaStream | null>(null);  // MediaStream object (needed to stop the camera)
    const [videoRef, setVideoRef] = useState<HTMLVideoElement | null>(null);  // Reference to <video> element
    const [capturedImage, setCapturedImage] = useState<File|  null>(null);   // Captured File object (shown in preview)
    const [cameraError, setCameraError] = useState<string | null>(null);    // UX: User-friendly error messages
    const [isLoadingCamera, setIsLoadingCamera] = useState(false);         // UX: Show loading spinner while requesting camera access
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);       // URL for previewing captured image
    const [isVideoReady, setIsVideoReady] = useState(false);                // Track if video metadata is loaded
    const [isProcessingBgRemoval, setIsProcessingBgRemoval] = useState(false); // Track background removal processing state
    const [bgRemovalError, setBgRemovalError] = useState<string | null>(null); // UX: Show if background removal fails

    const processAndSetFile = async (file: File) => {
        setIsProcessingBgRemoval(true);
        setBgRemovalError(null);
        setCaptureMode('preview'); // Switch to preview immediately
        
        try {
            console.log("Starting background removal...");
            
            // Configure to use CDN for assets to avoid 404s on Vercel
            // This fetches the WASM and ONNX files from img.ly's static CDN instead of local public folder
            const config = {
                publicPath: "https://staticimgly.com/@imgly/background-removal-data/1.7.0/dist/",
                debug: true
            };

            // 1. Remove Background
            const blob = await removeBackground(file, config);
            
            // 2. Create a new File object (strips EXIF data automatically)
            const processedFile = new File([blob], file.name, { type: 'image/png' });
            
            // 3. Create URL for preview
            const url = URL.createObjectURL(blob);
            
            // 4. Update all states to use the PROCESSED file
            
            // Update legacy states so existing upload logic works
            setSelectedFile(processedFile);
            setCapturedImage(processedFile);
            setPreviewUrl(url);
            
        } catch (error) {
            console.error('Background removal failed:', error);
            setBgRemovalError("Background removal failed. Using original image.");
            
            // Fallback: Use original file if removal fails
            setSelectedFile(file);
            setCapturedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
        } finally {
            setIsProcessingBgRemoval(false);
        }
    };
    

    // 1. Manage Stream Lifecycle (Stop tracks when stream changes or unmounts)
    useEffect(() => {
        return () => {
            if (stream) {
                console.log('Cleaning up stream tracks');
                stream.getTracks().forEach((track) => track.stop());
            }
        };
    }, [stream]);

    // 2. Handle Video Element Attachment & Dimension Checking
    useEffect(() => {
        let checkInterval: number | null = null;

        if (stream && videoRef) {
            console.log('=== Setting up video element with stream ===');
            console.log('Stream tracks:', stream.getTracks().map(t => `${t.kind}: ${t.label} (enabled: ${t.enabled}, readyState: ${t.readyState})`));

            videoRef.srcObject = stream;

            // Get actual camera dimensions from the MediaStream track
            const videoTrack = stream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            const capabilities = videoTrack.getCapabilities();
            console.log('Camera track settings:', settings);
            console.log('Camera track capabilities:', capabilities);

            let attempts = 0;
            
            // Poll for valid video dimensions
            const checkVideoDimensions = () => {
                attempts++;
                if (!videoRef) return;
                
                const width = videoRef.videoWidth;
                const height = videoRef.videoHeight;
                const readyState = videoRef.readyState;
                const rect = videoRef.getBoundingClientRect();

                console.log(`Attempt ${attempts}:`);
                console.log(`  - videoWidth/Height: ${width}x${height}`);
                console.log(`  - readyState: ${readyState}`);
                console.log(`  - paused: ${videoRef.paused}`);
                console.log(`  - ended: ${videoRef.ended}`);
                console.log(`  - rendered size: ${rect.width}x${rect.height}`);
                console.log(`  - srcObject: ${videoRef.srcObject ? 'present' : 'null'}`);

                // For capture purposes, we just need the stream to be active
                if (readyState >= 2 || attempts >= 10) { 
                    console.log(`✓ Video ready for capture (readyState: ${readyState})`);
                    setIsVideoReady(true);

                    if (checkInterval !== null) {
                        clearInterval(checkInterval);
                        checkInterval = null;
                    }
                }
            };

            // Start playing the video
            console.log('Calling video.play()...');
            videoRef.play()
                .then(() => {
                    console.log('✓ Video play() resolved successfully');
                    console.log(`Video state after play: paused=${videoRef.paused}, ended=${videoRef.ended}, readyState=${videoRef.readyState}`);
                })
                .catch((error) => {
                    console.error("❌ Error playing video:", error);
                    setCameraError('Failed to start video playback.');
                });

            // Check dimensions after a short delay
            setTimeout(checkVideoDimensions, 300);

            // Also check every 200ms until we're ready
            checkInterval = window.setInterval(checkVideoDimensions, 200);
        } else {
            // Reset video ready state when stream is stopped
            setIsVideoReady(false);
        }

        // Cleanup interval on unmount or dependency change
        return () => {
            if (checkInterval !== null) {
                clearInterval(checkInterval);
            }
        };
    }, [stream, videoRef]);

    // Cleanup blob URL when preview unmounts or changes
    useEffect(() => {
        return () => {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);
    
    const startCamera = async () => {
        try {
            setIsLoadingCamera(true);
            setCameraError(null);

            // 1: Request camera access with explicit constraints
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user', // Use front camera if available
                    width: { min: 640, ideal: 1280, max: 1920 },
                    height: { min: 480, ideal: 720, max: 1080 }
                },
            })

            // Log what we actually got
            const track = mediaStream.getVideoTracks()[0];
            const actualSettings = track.getSettings();
            console.log('Stream acquired with settings:', actualSettings);

            // CRITICAL: Store width/height NOW while they're available
            // Some browsers lose this info after the stream is created
            if (actualSettings.width && actualSettings.height) {
                console.log(`✓ Camera dimensions captured: ${actualSettings.width}x${actualSettings.height}`);
                // Store in the stream object as a custom property for later access
                (mediaStream as any)._capturedWidth = actualSettings.width;
                (mediaStream as any)._capturedHeight = actualSettings.height;
            }

            // 2: Store steam and switch to camera mode
            setStream(mediaStream);
            setCaptureMode('camera');

        } catch (error: any) {
            if (error.name === 'NotAllowedError') {
                setCameraError('Camera permission denied. Please allow camera access in your browser settings.');
                } else if (error.name === 'NotFoundError') {
                setCameraError('No camera found on this device.');
                } else if (error.name === 'NotReadableError') {
                setCameraError('Camera is already in use by another application.');
                } else {
                setCameraError('Could not access camera. Please try again.');
                }
            console.error('Error accessing camera:', error);
        } finally {
            setIsLoadingCamera(false);}
        };


    const capturePhoto = () => {
        if(!videoRef || !stream) {
            console.error('Video ref or stream is unavailable');
            return;
        }

        // Get dimensions from video element or fall back to stream track settings
        let captureWidth = videoRef.videoWidth;
        let captureHeight = videoRef.videoHeight;

        console.log(`Initial capture dimensions: ${captureWidth}x${captureHeight}`);

        // If video element dimensions are invalid, we'll try multiple fallback strategies
        if (captureWidth <= 2 || captureHeight <= 2) {
            // Strategy 1: Use captured dimensions from when stream was created
            const storedWidth = (stream as any)._capturedWidth;
            const storedHeight = (stream as any)._capturedHeight;

            if (storedWidth && storedHeight) {
                captureWidth = storedWidth;
                captureHeight = storedHeight;
                console.log(`Using stored camera dimensions: ${captureWidth}x${captureHeight}`);
            } else {
                // Strategy 2: Try track settings (may not work on all browsers)
                const videoTrack = stream.getVideoTracks()[0];
                const settings = videoTrack.getSettings();

                if (settings.width && settings.height) {
                    captureWidth = settings.width;
                    captureHeight = settings.height;
                    console.log(`Using track settings dimensions: ${captureWidth}x${captureHeight}`);
                } else {
                    // Strategy 3: Use the rendered size from DOM
                    const rect = videoRef.getBoundingClientRect();
                    if (rect.width > 100 && rect.height > 100) {
                        // Scale up to a reasonable resolution (3x the display size)
                        captureWidth = Math.round(rect.width * 3);
                        captureHeight = Math.round(rect.height * 3);
                        console.log(`Using rendered size (3x scale): ${captureWidth}x${captureHeight} (from ${rect.width}x${rect.height})`);
                    } else {
                        // Strategy 4: Use default dimensions as last resort
                        captureWidth = 1280;
                        captureHeight = 720;
                        console.warn(`Using default fallback dimensions: ${captureWidth}x${captureHeight}`);
                    }
                }
            }
        }

        // 1: Create a canvas to draw the current video frame
        const canvas = document.createElement('canvas');
        canvas.width = captureWidth;
        canvas.height = captureHeight;

        console.log(`Capturing photo at ${canvas.width}x${canvas.height}`);

        // 2: Draw the video frame onto the canvas
        const context = canvas.getContext('2d');
        if (!context) {
            console.error('Could not get canvas context');
            return;
        }

        // IMPORTANT: Video is mirrored (scaleX(-1)), but we want to capture UN-mirrored
        // So we need to flip the canvas horizontally to get the correct orientation
        context.save();
        context.scale(-1, 1);  // Flip horizontally
        context.drawImage(videoRef, -canvas.width, 0, canvas.width, canvas.height);
        context.restore();

        console.log('Canvas drawn (un-mirrored), converting to blob...');

        // 3: Convert the canvas to a Blob
        canvas.toBlob((blob) => {
            if (!blob) {
                console.error('Could not convert canvas to Blob');
                return;
            }

            console.log(`Blob created: ${blob.size} bytes, type: ${blob.type}`);

            // 4: Create a File object from the Blob
            const timestamp = Date.now();
            const file = new File(
                [blob],
                `skin-analysis-${timestamp}.jpg`,
                { type: 'image/jpeg' }
            );

            console.log(`File created: ${file.name}, size: ${file.size}`);

            // 5: Process the captured image (Remove Background)
            processAndSetFile(file);

            // 6: Stop the camera stream
            stopCamera();
        }, 'image/jpeg', 0.95); // High quality JPEG
    };
    
    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
    };

    // Handle file selection logic (shared by both upload methods)
    const handleFileSelection = (file: File) => {
        setSelectedFile(file);
    };

    const handleUploadCaptured = () => {
        if (!capturedImage) return;

        // Use the shared file selection logic
        handleFileSelection(capturedImage);
        
        // Automatically trigger upload
        // We need to set selectedFile first, but state updates are async
        // So we call a modified upload function directly with the file
        uploadFile(capturedImage);
    };

    // Handle file selection from file input
    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            processAndSetFile(event.target.files[0]);
        }
    }

    const uploadFile = async (file: File) => {
        setUploadStatus('Uploading...');
        setIsLoadingCamera(true); // Re-use this state to disable buttons

        // 1. Create FormData object to send file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userId);
        // Removed budget_max to default to None (Infinite)
        formData.append('bundle_mode', 'true');

        try {
            // 2. Send Post request to upload endpoint
            // Send to backend
            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await fetch(`${API_URL}/upload`, { 
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data: AnalysisResponse = await response.json();
                setUploadStatus('Upload was successful! ✅');
                // Pass the data up to the parent component
                onAnalysisComplete(data);
            } else {
                setUploadStatus('Upload failed. ❌');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            setUploadStatus('Network error ⚠️');
        } finally {
            setIsLoadingCamera(false);
        }
    };

    const handleUpload = () => {
        if (selectedFile) {
            uploadFile(selectedFile);
        } else {
            alert("Please select a file first.");
        }
    };

    return (
        <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-xl shadow-md space-y-4">
            {/* Error Message */}
            { (cameraError || bgRemovalError) && (
                <div className={`w-full p-3 mb-4 border rounded-md ${
                    cameraError ? 'bg-red-100 border-red-400 text-red-700' : 'bg-yellow-100 border-yellow-400 text-yellow-700'
                }`}>
                    <p className="text-sm">{cameraError || bgRemovalError}</p>
                    <button
                        onClick={() => { setCameraError(null); setBgRemovalError(null); }}
                        className="mt-2 text-xs underline hover:no-underline">Dismiss
                    </button>
                </div>
            )}

            {/* Mode: Camera */}
            { captureMode === 'camera' && (
                <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative w-full max-w-md h-[75vh] bg-black rounded-2xl overflow-hidden shadow-2xl border border-gray-800 flex flex-col">
                        <video
                            ref={setVideoRef}
                            autoPlay
                            playsInline
                            muted
                            style={{
                                transform: 'scaleX(-1)',
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                            }}
                            className="absolute inset-0"
                            onLoadedMetadata={(e) => {
                                const video = e.currentTarget;
                                console.log(`📹 Video metadata loaded: ${video.videoWidth}x${video.videoHeight}`);
                            }}
                        />
                        
                        {/* Camera Controls Overlay */}
                        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent flex justify-center items-center gap-8 pb-8">
                            <button
                                onClick={() => {
                                    stopCamera();
                                    setCaptureMode('file');
                                }}
                                className="p-4 rounded-full bg-gray-800/50 text-white backdrop-blur-sm hover:bg-gray-700/50 transition-all"
                            >
                                ✕
                            </button>
                            
                            <button
                                onClick={capturePhoto}
                                disabled={!isVideoReady}
                                className={`w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all transform active:scale-95 ${
                                    !isVideoReady ? 'opacity-50 cursor-not-allowed' : 'hover:bg-white/20'
                                }`}
                            >
                                <div className="w-16 h-16 bg-white rounded-full"></div>
                            </button>
                            
                            <div className="w-12"></div> {/* Spacer for balance */}
                        </div>
                    </div>
                </div>
            )}

            {/* Mode: Preview Captured/Selected Image */}
            { captureMode === 'preview' && (capturedImage || isProcessingBgRemoval) && (
                <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative w-full max-w-md h-[75vh] bg-black rounded-2xl overflow-hidden shadow-2xl border border-gray-800 flex flex-col">
                        
                        {/* Loading Overlay for Background Removal */}
                        {isProcessingBgRemoval && (
                            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm text-white">
                                <div className="animate-spin h-12 w-12 border-4 border-blue-500 border-t-transparent rounded-full mb-4"></div>
                                <p className="font-medium">Removing Background...</p>
                                <p className="text-xs text-gray-400 mt-2">Ensuring privacy & stripping metadata</p>
                            </div>
                        )}

                        <img
                            src={previewUrl || ''}
                            alt="Processed photo"
                            className="w-full h-full object-contain bg-[url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAMUlEQVQ4T2NkYGAQYcAP3uCTZhw1gGGYhAGBZIA/nYDCgBDAm9BGDWAAJyRCgLaBCAAgXwixzAS0pgAAAABJRU5ErkJggg==')] bg-repeat"
                        />
                        
                        {/* Preview Controls Overlay */}
                        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 via-black/50 to-transparent flex justify-between items-end">
                            <button 
                                onClick={() => {
                                    setCaptureMode('file');
                                    setCapturedImage(null);
                                    setSelectedFile(null);
                                }}
                                disabled={isProcessingBgRemoval}
                                className="px-6 py-3 rounded-full bg-gray-800/80 text-white font-medium backdrop-blur-sm hover:bg-gray-700 transition-all border border-gray-600 disabled:opacity-50"
                            >
                                ✕ Cancel
                            </button>
                            
                            <button 
                                onClick={handleUploadCaptured}
                                disabled={isLoadingCamera || isProcessingBgRemoval}
                                className={`px-6 py-3 rounded-full text-white font-bold shadow-lg transition-all transform flex items-center gap-2 ${
                                    isLoadingCamera || isProcessingBgRemoval
                                    ? 'bg-gray-500 cursor-wait' 
                                    : 'bg-blue-600 hover:bg-blue-500 hover:scale-105'
                                }`}
                            >
                                {isLoadingCamera ? (
                                    <>
                                        <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
                                        Uploading...
                                    </>
                                ) : (
                                    <>Analyze →</>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Mode: Initial File Upload Options */}
            { captureMode === 'file' && (
                <div className="flex flex-col items-center space-y-4">
                    <button
                        onClick={startCamera}
                        className="w-full py-2 px-4 rounded-md text-white font-medium bg-blue-600 hover:bg-blue-700"
                        disabled={isLoadingCamera}
                    >
                        {isLoadingCamera ? 'Starting Camera...' : 'Use Camera to Capture Photo'}
                    </button>
                </div>
            )}

            {/* File Upload Section - Only show in 'file' mode */}
            {captureMode === 'file' && (
                <>
                    <h2 className="text-xl font-bold text-gray-800">Upload Image</h2>

                    <div className="flex items-center justify-center w-full">
                {/* File Input */}
                <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span></p>
                    </div>
                    <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleFileSelect}
                    />

                </label>
            </div>

            {selectedFile && (
                <p className="text-sm text-gray-600">Selected file: {selectedFile.name}</p>
            )}
            <button
                onClick={handleUpload}
                disabled={!selectedFile}
                className={`w-full py-2 px-4 rounded-md text-white font-medium transition-colors ${
                    !selectedFile
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
            >
                Upload for Analysis
            </button>

            {uploadStatus && (
                <div className={`p-3 rounded-md text-sm ${
                    uploadStatus.includes('✅') ? 'bg-green-100 text-green-700' :
                    uploadStatus.includes('❌') ? 'bg-red-100 text-red-700' :
                    'bg-blue-100 text-blue-700'
                }`}>
                    {uploadStatus}
                </div>
            )}
                </>
            )}
        </div>
    );
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
