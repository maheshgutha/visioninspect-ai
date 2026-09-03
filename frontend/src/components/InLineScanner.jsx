import React, { useState, useEffect, useRef } from "react";
import api, { resolveImageUrl } from "../api/axios";
import DefectResultCard from "./DefectResultCard";
import QualityReportPanel from "./QualityReportPanel";

export default function InLineScanner() {
  const [mode, setMode] = useState("camera"); // "camera" | "upload"
  const [productName, setProductName] = useState("Aluminum Bracket - Model A2");
  const [batchNumber, setBatchNumber] = useState("BATCH-2026-LINE-01");

  // Camera stream state
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  // Scanner state
  const [isScanning, setIsScanning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [latestInspection, setLatestInspection] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [error, setError] = useState("");

  // Batch upload state
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchIndex, setBatchIndex] = useState(0);
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);

  const autoScanTimerRef = useRef(null);
  const streamRef = useRef(null);

  // Initialize camera devices list
  useEffect(() => {
    async function getDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setVideoDevices(videoInputs);
        if (videoInputs.length > 0) {
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      } catch (err) {
        console.warn("Could not enumerate camera devices:", err);
      }
    }
    getDevices();

    return () => {
      stopCamera();
      if (autoScanTimerRef.current) clearInterval(autoScanTimerRef.current);
    };
  }, []);

  // Start real device camera feed
  const startCamera = async (deviceId = selectedDeviceId) => {
    setCameraError("");
    stopCamera();

    try {
      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsCameraActive(true);
    } catch (err) {
      console.error("Camera access error:", err);
      setCameraError(
        "Could not access camera. Please allow camera permissions in your browser or select an alternate video input."
      );
      setIsCameraActive(false);
    }
  };

  // Stop camera feed
  const stopCamera = () => {
    if (autoScanTimerRef.current) {
      clearInterval(autoScanTimerRef.current);
      setIsScanning(false);
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  // Capture frame blob from real camera video element
  const captureCameraFrameBlob = () => {
    return new Promise((resolve, reject) => {
      if (!videoRef.current || !isCameraActive) {
        return reject(new Error("Camera stream is not active. Please start camera first."));
      }

      const video = videoRef.current;
      const canvas = canvasRef.current || document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        if (!blob) return reject(new Error("Failed to capture image frame from video feed."));
        const file = new File([blob], `camera_scan_${Date.now()}.png`, { type: "image/png" });
        resolve(file);
      }, "image/png");
    });
  };

  // Send captured frame/file to backend API
  const processImageFile = async (fileToUpload, name, batch) => {
    const formData = new FormData();
    formData.append("product_name", name || productName);
    formData.append("batch_number", batch || batchNumber);
    formData.append("notes", "Live Camera In-Line Scan");
    formData.append("file", fileToUpload);

    const res = await api.post("/api/inspections/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  };

  // Run single camera scan cycle
  const runCameraScanCycle = async (overrideFile = null) => {
    setLoading(true);
    setError("");

    try {
      let fileToUpload = overrideFile;
      if (!fileToUpload) {
        fileToUpload = await captureCameraFrameBlob();
      }

      const result = await processImageFile(fileToUpload);
      setLatestInspection(result);
      setScanHistory((prev) => [result, ...prev.slice(0, 19)]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "In-line scanning process failed.");
    } finally {
      setLoading(false);
    }
  };

  // Toggle Auto Camera Scanner
  const toggleAutoScanner = () => {
    if (!isCameraActive) {
      setError("Please start the camera stream first before enabling auto-scanning.");
      return;
    }

    if (isScanning) {
      if (autoScanTimerRef.current) clearInterval(autoScanTimerRef.current);
      setIsScanning(false);
    } else {
      setIsScanning(true);
      runCameraScanCycle();
      autoScanTimerRef.current = setInterval(() => {
        runCameraScanCycle();
      }, 3500); // scan frame every 3.5 seconds
    }
  };

  // Batch Multi-File Upload & Process
  const handleBatchSelect = (e) => {
    const files = Array.from(e.target.files);
    setBatchFiles(files);
    setBatchIndex(0);
  };

  const startBatchProcess = async () => {
    if (batchFiles.length === 0) {
      setError("Please select one or more product image files for batch scanning.");
      return;
    }

    setIsProcessingBatch(true);
    setError("");

    for (let i = 0; i < batchFiles.length; i++) {
      setBatchIndex(i + 1);
      try {
        const result = await processImageFile(batchFiles[i], `${productName} #${i + 1}`);
        setLatestInspection(result);
        setScanHistory((prev) => [result, ...prev.slice(0, 19)]);
        await new Promise((r) => setTimeout(r, 600));
      } catch (err) {
        console.error("Batch processing error on item", i, err);
      }
    }

    setIsProcessingBatch(false);
  };

  return (
    <div className="space-y-6">
      {/* Hidden offscreen canvas for capturing camera video frames */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Top Banner & Mode Selector */}
      <div className="bg-slate-900 text-white rounded-xl p-5 shadow-sm border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">📹</span>
            <h3 className="text-lg font-bold">Real Camera In-Line Conveyor Scanner</h3>
          </div>
          <p className="text-xs text-slate-400">
            Real WebRTC camera feed acquisition, live conveyor optical inspection, defect detection, and severity analysis.
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-800 p-1 rounded-lg border border-slate-700 self-start md:self-auto">
          <button
            onClick={() => setMode("camera")}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              mode === "camera" ? "bg-brand-600 text-white shadow-sm" : "text-slate-400 hover:text-white"
            }`}
          >
            📹 Live Camera Scanner
          </button>
          <button
            onClick={() => {
              stopCamera();
              setMode("upload");
            }}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              mode === "upload" ? "bg-brand-600 text-white shadow-sm" : "text-slate-400 hover:text-white"
            }`}
          >
            📦 Batch File Upload
          </button>
        </div>
      </div>

      {cameraError && (
        <div className="bg-red-50 text-red-700 text-xs font-semibold rounded-lg p-3.5 border border-red-200">
          ⚠️ {cameraError}
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 text-xs font-semibold rounded-lg p-3.5 border border-red-200">
          ⚠️ {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Video Camera Viewport & Controls */}
        <div className="lg:col-span-6 space-y-6">
          {mode === "camera" ? (
            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-sm relative">
              {/* Viewport Header */}
              <div className="p-3 bg-slate-800/80 border-b border-slate-700 flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200 flex items-center gap-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      isCameraActive ? (isScanning ? "bg-emerald-400 animate-ping" : "bg-emerald-400") : "bg-red-500"
                    }`}
                  ></span>
                  {isCameraActive ? "LIVE CAMERA STREAM ACTIVE" : "CAMERA DISCONNECTED"}
                </span>

                {videoDevices.length > 1 && (
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => {
                      setSelectedDeviceId(e.target.value);
                      if (isCameraActive) startCamera(e.target.value);
                    }}
                    className="bg-slate-900 border border-slate-700 text-slate-300 text-[11px] rounded px-2 py-0.5 outline-none"
                  >
                    {videoDevices.map((dev, i) => (
                      <option key={dev.deviceId} value={dev.deviceId}>
                        {dev.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Real Video Element Viewport */}
              <div className="relative h-72 bg-slate-950 flex items-center justify-center overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-contain bg-slate-950 ${!isCameraActive ? "hidden" : ""}`}
                />

                {!isCameraActive && (
                  <div className="text-center p-6 text-slate-500">
                    <span className="text-4xl block mb-2">📷</span>
                    <p className="text-xs text-slate-400 mb-3">Click below to start live video stream from your camera.</p>
                    <button
                      onClick={() => startCamera()}
                      className="bg-brand-600 hover:bg-brand-700 text-white font-bold px-4 py-2 rounded-lg text-xs transition-colors shadow-md"
                    >
                      Start Camera Feed
                    </button>
                  </div>
                )}

                {/* Laser Scanning Line Animation when active */}
                {isCameraActive && (isScanning || loading) && (
                  <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#22d3ee] animate-pulse top-1/2 -translate-y-1/2"></div>
                )}

                {/* Latest Scan Result Overlay */}
                {latestInspection && isCameraActive && (
                  <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between bg-slate-900/90 backdrop-blur-sm p-2 rounded-lg border border-slate-700 text-xs">
                    <span className="font-semibold text-white truncate max-w-[180px]">{latestInspection.product_name}</span>
                    <span
                      className={`font-bold px-2.5 py-0.5 rounded-full text-[10px] ${
                        latestInspection.status === "pass"
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                          : "bg-red-500/20 text-red-300 border border-red-500/40"
                      }`}
                    >
                      {latestInspection.status.toUpperCase()} (SEV {latestInspection.severity_score || 0})
                    </span>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-4 bg-slate-900 border-t border-slate-800 space-y-3">
                <div className="flex items-center gap-3">
                  {isCameraActive ? (
                    <>
                      <button
                        onClick={toggleAutoScanner}
                        className={`flex-1 font-bold py-2.5 rounded-lg text-xs transition-all flex items-center justify-center gap-2 ${
                          isScanning
                            ? "bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-600/20"
                            : "bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg shadow-emerald-600/20"
                        }`}
                      >
                        <span>{isScanning ? "⏸️ Pause Conveyor Scanner" : "▶️ Start Auto-Camera Scanning"}</span>
                      </button>

                      <button
                        onClick={() => runCameraScanCycle()}
                        disabled={loading || isScanning}
                        className="bg-brand-600 hover:bg-brand-700 text-white text-xs font-bold px-4 py-2.5 rounded-lg transition-colors disabled:opacity-50 shadow-md"
                      >
                        {loading ? "Scanning..." : "📸 Snap Photo & Inspect"}
                      </button>

                      <button
                        onClick={stopCamera}
                        className="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold px-3 py-2.5 rounded-lg border border-slate-700"
                      >
                        Stop
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => startCamera()}
                      className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 rounded-lg text-xs transition-all shadow-md"
                    >
                      📷 Enable Live Camera Stream
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Batch Upload Mode */
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <h4 className="font-bold text-slate-800 text-sm">Batch Image Upload & Sequential Scan</h4>
              <div>
                <label className="text-xs text-slate-600 font-medium block mb-1">
                  Select Product Image Files (Multi-Select)
                </label>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleBatchSelect}
                  className="w-full text-xs text-slate-600 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-brand-50 file:text-brand-700 file:font-semibold"
                />
                {batchFiles.length > 0 && (
                  <p className="text-xs text-emerald-600 mt-1 font-semibold">
                    {batchFiles.length} file(s) ready for batch scanning.
                  </p>
                )}
              </div>

              {isProcessingBatch && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-500 font-medium">
                    <span>Processing Batch Queue...</span>
                    <span>{batchIndex} / {batchFiles.length}</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-brand-600 h-full transition-all duration-300"
                      style={{ width: `${(batchIndex / batchFiles.length) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}

              <button
                onClick={startBatchProcess}
                disabled={isProcessingBatch || batchFiles.length === 0}
                className="w-full bg-brand-600 hover:bg-brand-700 text-white font-bold py-2.5 rounded-lg text-xs transition-all disabled:opacity-50 shadow-md"
              >
                {isProcessingBatch ? `Processing Item ${batchIndex}...` : `🚀 Start Batch Processing (${batchFiles.length} files)`}
              </button>
            </div>
          )}

          {/* Scanner Product Settings */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wide">Inspection Target Label</h4>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="text-slate-600 font-medium block mb-1">Product Name</label>
                <input
                  type="text"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-2.5 py-1.5 bg-slate-50 focus:bg-white focus:ring-1 focus:ring-brand-500 outline-none"
                />
              </div>
              <div>
                <label className="text-slate-600 font-medium block mb-1">Batch / Shift ID</label>
                <input
                  type="text"
                  value={batchNumber}
                  onChange={(e) => setBatchNumber(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-2.5 py-1.5 bg-slate-50 focus:bg-white focus:ring-1 focus:ring-brand-500 outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Real-Time Results & History Log */}
        <div className="lg:col-span-6 space-y-6">
          {latestInspection ? (
            <div className="space-y-4">
              <DefectResultCard inspection={latestInspection} />
              {latestInspection.quality_report && (
                <QualityReportPanel report={latestInspection.quality_report} />
              )}
            </div>
          ) : (
            <div className="bg-slate-50 rounded-xl border border-dashed border-slate-300 p-12 text-center text-slate-400 text-xs">
              <span className="text-3xl block mb-2">⚡</span>
              No camera scans performed yet. Enable live camera feed and snap a photo or start auto-scanning.
            </div>
          )}

          {/* Real Scan History Log */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
            <div className="p-4 border-b border-slate-100 flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wide">Live Stream Inspection Log</h4>
              <span className="text-[11px] text-slate-400">{scanHistory.length} Scanned</span>
            </div>

            <div className="max-h-60 overflow-y-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 text-slate-500 text-[10px] uppercase font-semibold border-b border-slate-200 sticky top-0">
                    <th className="py-2 px-3">Time</th>
                    <th className="py-2 px-3">Product</th>
                    <th className="py-2 px-3">Status</th>
                    <th className="py-2 px-3">Severity</th>
                    <th className="py-2 px-3">Defect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {scanHistory.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="py-2 px-3 text-slate-400 text-[11px]">
                        {new Date(item.created_at).toLocaleTimeString()}
                      </td>
                      <td className="py-2 px-3 font-semibold text-slate-800 truncate max-w-[120px]">
                        {item.product_name}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`font-bold px-2 py-0.5 rounded-full text-[10px] ${
                            item.status === "pass"
                              ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              : "bg-red-50 text-red-700 border border-red-200"
                          }`}
                        >
                          {item.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-semibold text-slate-700">
                        {item.severity_score !== null ? `${item.severity_score} (${item.severity_level || "Low"})` : "-"}
                      </td>
                      <td className="py-2 px-3 text-slate-600 capitalize text-[11px]">
                        {item.defect_type ? item.defect_type.replace(/_/g, " ") : "-"}
                      </td>
                    </tr>
                  ))}
                  {scanHistory.length === 0 && (
                    <tr>
                      <td colSpan="5" className="py-6 text-center text-slate-400 text-xs">
                        No live camera scans recorded yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
