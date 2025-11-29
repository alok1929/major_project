"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface PersonResult {
  person_id: number;
  is_correct: boolean;
  confidence: number;
  current_angle: number;
  state: string;
  current_arm: string;
  rep_count: number;
  exercise_complete: boolean;
  grades: {
    right: Record<string, number>;
    left: Record<string, number>;
  };
}

interface RepComplete {
  person_id: number;
  arm: string;
  rep_number: number;
  grade: string;
  correct_ratio: number;
  peak_angle: number;
  likely_issue: string | null;
}

interface LLMFeedback {
  person_id: number;
  is_final: boolean;
  feedback: string;
}

interface FrameResult {
  type: string;
  frame_count: number;
  persons: PersonResult[];
}

export default function ExerciseTracker() {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Results state
  const [persons, setPersons] = useState<PersonResult[]>([]);
  const [repHistory, setRepHistory] = useState<RepComplete[]>([]);
  const [feedbackMessages, setFeedbackMessages] = useState<LLMFeedback[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize webcam
  const initWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  }, []);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    
    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
    };
    
    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      setIsStreaming(false);
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case "frame_result":
          setPersons(data.persons);
          setFrameCount(data.frame_count);
          break;
          
        case "rep_complete":
          setRepHistory((prev) => [...prev, data as RepComplete]);
          break;
          
        case "llm_feedback":
          setFeedbackMessages((prev) => [...prev, data as LLMFeedback]);
          break;
          
        case "arm_switch":
          console.log(`Person ${data.person_id} switched to ${data.new_arm} arm`);
          break;
          
        case "exercise_complete":
          console.log(`Person ${data.person_id} completed exercise`);
          break;
          
        case "session_summary":
          console.log("Session summary:", data);
          break;
          
        case "error":
          console.error("Server error:", data.message);
          break;
          
        default:
          console.log("Unknown message type:", data);
      }
    };
    
    wsRef.current = ws;
  }, []);

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    
    if (!ctx) return;
    
    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64 JPEG
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const base64Data = dataUrl.split(",")[1];
    
    // Send to server
    wsRef.current.send(JSON.stringify({
      type: "frame",
      data: base64Data
    }));
  }, []);

  // Start streaming
  const startStreaming = useCallback(() => {
    if (!isConnected) return;
    
    setIsStreaming(true);
    
    // Send frames at ~10 FPS
    streamIntervalRef.current = setInterval(() => {
      captureAndSendFrame();
    }, 100);
  }, [isConnected, captureAndSendFrame]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    
    // Request summary
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "get_summary" }));
    }
  }, []);

  // Reset session
  const resetSession = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
    
    setPersons([]);
    setRepHistory([]);
    setFeedbackMessages([]);
    setFrameCount(0);
  }, []);

  // Initialize on mount
  useEffect(() => {
    initWebcam();
    connectWebSocket();
    
    return () => {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [initWebcam, connectWebSocket]);

  // Get grade color
  const getGradeColor = (grade: string) => {
    switch (grade) {
      case "Excellent": return "text-green-500";
      case "Good": return "text-blue-500";
      case "Needs Improvement": return "text-yellow-500";
      case "Poor": return "text-red-500";
      default: return "text-gray-500";
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">🏋️ Exercise Tracker</h1>
        
        {/* Status Bar */}
        <div className="flex items-center gap-4 mb-6">
          <div className={`px-3 py-1 rounded-full text-sm ${isConnected ? "bg-green-600" : "bg-red-600"}`}>
            {isConnected ? "Connected" : "Disconnected"}
          </div>
          <div className="text-gray-400">
            Frame: {frameCount}
          </div>
        </div>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">📹 Camera Feed</h2>
              
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full rounded-lg bg-black"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {/* Overlay for person indicators */}
                {persons.map((person) => (
                  <div
                    key={person.person_id}
                    className={`absolute top-2 ${person.person_id === 0 ? "left-2" : "right-2"} 
                      p-2 rounded-lg ${person.is_correct ? "bg-green-600/80" : "bg-red-600/80"}`}
                  >
                    <div className="text-sm font-bold">Person {person.person_id}</div>
                    <div className="text-xs">
                      {person.current_arm.toUpperCase()} arm | Rep {person.rep_count}/5
                    </div>
                    <div className="text-xs">
                      Angle: {person.current_angle.toFixed(1)}° | {person.state}
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Controls */}
              <div className="flex gap-4 mt-4">
                <button
                  onClick={isStreaming ? stopStreaming : startStreaming}
                  disabled={!isConnected}
                  className={`px-6 py-2 rounded-lg font-semibold ${
                    isStreaming
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-green-600 hover:bg-green-700"
                  } disabled:bg-gray-600 disabled:cursor-not-allowed`}
                >
                  {isStreaming ? "⏹ Stop" : "▶ Start"}
                </button>
                
                <button
                  onClick={resetSession}
                  className="px-6 py-2 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700"
                >
                  🔄 Reset
                </button>
              </div>
            </div>
            
            {/* Rep History */}
            <div className="bg-gray-800 rounded-lg p-4 mt-6">
              <h2 className="text-xl font-semibold mb-4">📊 Rep History</h2>
              
              <div className="max-h-60 overflow-y-auto space-y-2">
                {repHistory.length === 0 ? (
                  <p className="text-gray-400">No reps completed yet</p>
                ) : (
                  repHistory.map((rep, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-700 rounded-lg p-3"
                    >
                      <div>
                        <span className="font-semibold">Person {rep.person_id}</span>
                        <span className="text-gray-400 ml-2">
                          {rep.arm.toUpperCase()} arm - Rep {rep.rep_number}
                        </span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className={`font-bold ${getGradeColor(rep.grade)}`}>
                          {rep.grade}
                        </span>
                        <span className="text-gray-400">
                          {(rep.correct_ratio * 100).toFixed(0)}%
                        </span>
                        <span className="text-gray-400">
                          Peak: {rep.peak_angle.toFixed(1)}°
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
          
          {/* Sidebar - Person Stats & Feedback */}
          <div className="space-y-6">
            
            {/* Person Stats */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">👥 Current Stats</h2>
              
              {persons.length === 0 ? (
                <p className="text-gray-400">No persons detected</p>
              ) : (
                persons.map((person) => (
                  <div key={person.person_id} className="mb-4 p-3 bg-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-bold">Person {person.person_id}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        person.is_correct ? "bg-green-600" : "bg-red-600"
                      }`}>
                        {person.is_correct ? "Correct" : "Incorrect"}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-400">Arm:</span>
                        <span className="ml-2">{person.current_arm.toUpperCase()}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Rep:</span>
                        <span className="ml-2">{person.rep_count}/5</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Angle:</span>
                        <span className="ml-2">{person.current_angle.toFixed(1)}°</span>
                      </div>
                      <div>
                        <span className="text-gray-400">State:</span>
                        <span className="ml-2">{person.state}</span>
                      </div>
                    </div>
                    
                    {/* Grades */}
                    <div className="mt-2 text-xs">
                      <div className="text-gray-400">Right arm grades:</div>
                      <div className="flex gap-2 mt-1">
                        <span className="text-green-400">E:{person.grades.right.Excellent}</span>
                        <span className="text-blue-400">G:{person.grades.right.Good}</span>
                        <span className="text-yellow-400">NI:{person.grades.right["Needs Improvement"]}</span>
                        <span className="text-red-400">P:{person.grades.right.Poor}</span>
                      </div>
                    </div>
                    
                    {person.exercise_complete && (
                      <div className="mt-2 text-green-400 font-bold">
                        ✅ Exercise Complete!
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            
            {/* LLM Feedback */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">🤖 AI Feedback</h2>
              
              <div className="max-h-96 overflow-y-auto space-y-4">
                {feedbackMessages.length === 0 ? (
                  <p className="text-gray-400">
                    Feedback will appear after completing reps...
                  </p>
                ) : (
                  feedbackMessages.map((feedback, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg ${
                        feedback.is_final ? "bg-green-900/50 border border-green-600" : "bg-gray-700"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-bold">Person {feedback.person_id}</span>
                        {feedback.is_final && (
                          <span className="px-2 py-0.5 bg-green-600 rounded text-xs">
                            Final
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-300 whitespace-pre-wrap">
                        {feedback.feedback}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}