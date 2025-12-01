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
  feedback_json?: {
    session_summary?: string;
    what_went_well?: string;
    areas_to_improve?: string;
    tips_for_next_session?: string;
    focus_point?: string;
    raw_feedback?: string;
  };
  feedback_type?: "periodic" | "final";
  rep_count?: number;
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
  const [progressSaved, setProgressSaved] = useState(false);
  
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
          
        case "progress_saved":
          console.log("Progress saved:", data.message);
          setProgressSaved(true);
          // Stop streaming automatically
          setIsStreaming(false);
          if (streamIntervalRef.current) {
            clearInterval(streamIntervalRef.current);
            streamIntervalRef.current = null;
          }
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
    setProgressSaved(false);
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Success Message Banner */}
      {progressSaved && (
        <div className="sticky top-0 z-[60] bg-gradient-to-r from-green-600 to-green-500 text-white shadow-lg">
          <div className="max-w-[1920px] mx-auto px-6 py-4">
            <div className="flex items-center justify-center gap-3">
              <span className="text-2xl">✅</span>
              <span className="text-lg font-semibold">Progress saved successfully</span>
            </div>
          </div>
        </div>
      )}
      
      {/* Header Bar - Fixed at top */}
      <div className={`sticky ${progressSaved ? 'top-[60px]' : 'top-0'} z-50 bg-gray-900/95 backdrop-blur-sm border-b border-gray-700`}>
        <div className="max-w-[1920px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                🏋️ Exercise Tracker
              </h1>
              <div className="flex items-center gap-3">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                  isConnected ? "bg-green-500/20 text-green-400 border border-green-500/30" : "bg-red-500/20 text-red-400 border border-red-500/30"
                }`}>
                  <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
                  {isConnected ? "Connected" : "Disconnected"}
                </div>
                <div className="text-sm text-gray-400">
                  Frame: <span className="text-gray-300 font-mono">{frameCount}</span>
                </div>
              </div>
            </div>
            
            {/* Controls */}
            <div className="flex items-center gap-3">
              <button
                onClick={isStreaming ? stopStreaming : startStreaming}
                disabled={!isConnected || progressSaved}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold transition-all duration-200 ${
                  isStreaming
                    ? "bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/30"
                    : "bg-green-500 hover:bg-green-600 text-white shadow-lg shadow-green-500/30"
                } disabled:bg-gray-700 disabled:cursor-not-allowed disabled:shadow-none`}
              >
                {isStreaming ? (
                  <>
                    <span>⏹</span> Stop
                  </>
                ) : (
                  <>
                    <span>▶</span> Start
                  </>
                )}
              </button>
              
              <button
                onClick={resetSession}
                className="flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 text-white transition-all duration-200"
              >
                <span>🔄</span> Reset
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Side by side layout */}
      <div className={`flex ${progressSaved ? 'h-[calc(100vh-140px)]' : 'h-[calc(100vh-80px)]'} max-w-[1920px] mx-auto`}>
        {/* Main Video Area - Takes 70% width */}
        <div className="flex-1 flex flex-col p-6 overflow-hidden">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6 h-full flex flex-col shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-200">📷 Camera Feed</h2>
              {persons.length > 0 && (
                <div className="flex gap-2">
                  {persons.map((person) => (
                    <div
                      key={person.person_id}
                      className={`px-3 py-1 rounded-full text-xs font-medium ${
                        person.is_correct 
                          ? "bg-green-500/20 text-green-400 border border-green-500/30" 
                          : "bg-red-500/20 text-red-400 border border-red-500/30"
                      }`}
                    >
                      Person {person.person_id}: {person.is_correct ? "✓" : "✗"}
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Video Container - Large and prominent */}
            <div className="relative flex-1 rounded-xl overflow-hidden bg-black shadow-inner border border-gray-700/50">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {/* Overlay indicators - Bottom right corner */}
              {persons.map((person) => (
                <div
                  key={person.person_id}
                  className={`absolute bottom-4 ${person.person_id === 0 ? "left-4" : "right-4"} 
                    bg-gray-900/90 backdrop-blur-sm p-4 rounded-xl border ${
                      person.is_correct ? "border-green-500/50" : "border-red-500/50"
                    } shadow-xl`}
                >
                  <div className="space-y-1">
                    <div className="text-sm font-bold text-white">Person {person.person_id}</div>
                    <div className="text-xs text-gray-300">
                      {person.current_arm.toUpperCase()} arm • Rep {person.rep_count}/5
                    </div>
                    <div className="text-xs text-gray-400">
                      Angle: <span className="text-gray-200 font-mono">{person.current_angle.toFixed(1)}°</span>
                    </div>
                    <div className="text-xs text-gray-400 capitalize">
                      State: <span className="text-gray-200">{person.state}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar - Fixed width, scrollable */}
        <div className="w-[420px] bg-gray-900/50 border-l border-gray-700/50 overflow-y-auto">
          <div className="p-6 space-y-6">
            
            {/* Current Stats Card */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-5 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-200 flex items-center gap-2">
                <span>📊</span> Current Stats
              </h2>
              
              {persons.length === 0 ? (
                <p className="text-gray-400 text-sm">No persons detected</p>
              ) : (
                <div className="space-y-4">
                  {persons.map((person) => (
                    <div key={person.person_id} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600/30">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-semibold text-white">Person {person.person_id}</span>
                        <span className={`px-2.5 py-1 rounded-md text-xs font-medium ${
                          person.is_correct 
                            ? "bg-green-500/20 text-green-400 border border-green-500/30" 
                            : "bg-red-500/20 text-red-400 border border-red-500/30"
                        }`}>
                          {person.is_correct ? "✓ Correct" : "✗ Incorrect"}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3 text-sm mb-3">
                        <div className="flex flex-col">
                          <span className="text-gray-400 text-xs mb-1">Arm</span>
                          <span className="text-white font-medium">{person.current_arm.toUpperCase()}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-gray-400 text-xs mb-1">Rep</span>
                          <span className="text-white font-medium">{person.rep_count}/5</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-gray-400 text-xs mb-1">Angle</span>
                          <span className="text-white font-mono font-medium">{person.current_angle.toFixed(1)}°</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-gray-400 text-xs mb-1">State</span>
                          <span className="text-white capitalize font-medium">{person.state}</span>
                        </div>
                      </div>
                      
                      {/* Grades - Visual indicators */}
                      <div className="mt-3 pt-3 border-t border-gray-600/30">
                        <div className="text-xs text-gray-400 mb-2">Right Arm Grades</div>
                        <div className="flex gap-2">
                          <div className="flex-1 bg-green-500/10 rounded px-2 py-1 text-center">
                            <div className="text-green-400 font-bold text-xs">{person.grades.right.Excellent || 0}</div>
                            <div className="text-[10px] text-gray-500">E</div>
                          </div>
                          <div className="flex-1 bg-blue-500/10 rounded px-2 py-1 text-center">
                            <div className="text-blue-400 font-bold text-xs">{person.grades.right.Good || 0}</div>
                            <div className="text-[10px] text-gray-500">G</div>
                          </div>
                          <div className="flex-1 bg-yellow-500/10 rounded px-2 py-1 text-center">
                            <div className="text-yellow-400 font-bold text-xs">{person.grades.right["Needs Improvement"] || 0}</div>
                            <div className="text-[10px] text-gray-500">NI</div>
                          </div>
                          <div className="flex-1 bg-red-500/10 rounded px-2 py-1 text-center">
                            <div className="text-red-400 font-bold text-xs">{person.grades.right.Poor || 0}</div>
                            <div className="text-[10px] text-gray-500">P</div>
                          </div>
                        </div>
                      </div>
                      
                      {person.exercise_complete && (
                        <div className="mt-3 pt-3 border-t border-green-500/30">
                          <div className="flex items-center gap-2 text-green-400 font-semibold text-sm">
                            <span>✅</span> Exercise Complete!
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Rep History Card */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-5 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-200 flex items-center gap-2">
                <span>📈</span> Rep History
              </h2>
              
              <div className="max-h-64 overflow-y-auto space-y-2 custom-scrollbar">
                {repHistory.length === 0 ? (
                  <p className="text-gray-400 text-sm text-center py-4">No reps completed yet</p>
                ) : (
                  repHistory.map((rep, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-700/50 rounded-lg p-3 border border-gray-600/30 hover:bg-gray-700/70 transition-colors"
                    >
                      <div className="flex flex-col">
                        <div className="text-sm font-medium text-white">
                          {rep.arm.toUpperCase()} arm • Rep {rep.rep_number}
                        </div>
                        <div className="text-xs text-gray-400">
                          Peak: {rep.peak_angle.toFixed(1)}°
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`font-bold text-sm ${getGradeColor(rep.grade)}`}>
                          {rep.grade}
                        </span>
                        <div className="text-xs text-gray-400 bg-gray-600/50 px-2 py-1 rounded">
                          {(rep.correct_ratio * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* AI Feedback Card */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-5 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-200 flex items-center gap-2">
                <span>🤖</span> AI Feedback
              </h2>
              
              <div className="space-y-4 max-h-[600px] overflow-y-auto custom-scrollbar">
                {feedbackMessages.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-gray-400 text-sm mb-2">Feedback will appear here</p>
                    <p className="text-gray-500 text-xs">Complete reps to receive AI feedback</p>
                  </div>
                ) : (
                  feedbackMessages.map((feedback, index) => {
                    const isFinal = feedback.is_final || feedback.feedback_type === "final";
                    const feedbackJson = feedback.feedback_json;
                    
                    return (
                      <div
                        key={index}
                        className={`rounded-xl p-4 border transition-all duration-200 ${
                          isFinal
                            ? "bg-gradient-to-br from-green-900/30 to-green-800/20 border-green-500/50 shadow-lg shadow-green-500/10"
                            : "bg-gray-700/50 border-gray-600/30 hover:bg-gray-700/70"
                        }`}
                      >
                        {/* Header */}
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-white">Person {feedback.person_id}</span>
                            {feedback.rep_count && (
                              <span className="text-xs text-gray-400">
                                After {feedback.rep_count} reps
                              </span>
                            )}
                          </div>
                          <span className={`px-2.5 py-1 rounded-md text-xs font-medium ${
                            isFinal
                              ? "bg-green-500/30 text-green-300 border border-green-500/50"
                              : "bg-blue-500/30 text-blue-300 border border-blue-500/50"
                          }`}>
                            {isFinal ? "Final" : "Progress"}
                          </span>
                        </div>

                        {/* Structured Feedback */}
                        {feedbackJson ? (
                          <div className="space-y-3 text-sm">
                            {feedbackJson.session_summary && (
                              <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-600/30">
                                <div className="text-xs font-semibold text-gray-300 mb-1 flex items-center gap-1">
                                  <span>📊</span> Session Summary
                                </div>
                                <p className="text-gray-200 text-xs leading-relaxed">{feedbackJson.session_summary}</p>
                              </div>
                            )}
                            
                            {feedbackJson.what_went_well && (
                              <div className="bg-green-500/10 rounded-lg p-3 border border-green-500/20">
                                <div className="text-xs font-semibold text-green-400 mb-1 flex items-center gap-1">
                                  <span>✅</span> What You Did Well
                                </div>
                                <p className="text-gray-200 text-xs leading-relaxed">{feedbackJson.what_went_well}</p>
                              </div>
                            )}
                            
                            {feedbackJson.areas_to_improve && (
                              <div className="bg-yellow-500/10 rounded-lg p-3 border border-yellow-500/20">
                                <div className="text-xs font-semibold text-yellow-400 mb-1 flex items-center gap-1">
                                  <span>⚠️</span> Areas to Improve
                                </div>
                                <p className="text-gray-200 text-xs leading-relaxed">{feedbackJson.areas_to_improve}</p>
                              </div>
                            )}
                            
                            {feedbackJson.tips_for_next_session && (
                              <div className="bg-blue-500/10 rounded-lg p-3 border border-blue-500/20">
                                <div className="text-xs font-semibold text-blue-400 mb-1 flex items-center gap-1">
                                  <span>💡</span> Tips for Next Session
                                </div>
                                <p className="text-gray-200 text-xs leading-relaxed">{feedbackJson.tips_for_next_session}</p>
                              </div>
                            )}
                            
                            {feedbackJson.focus_point && (
                              <div className="bg-purple-500/10 rounded-lg p-3 border border-purple-500/20">
                                <div className="text-xs font-semibold text-purple-400 mb-1 flex items-center gap-1">
                                  <span>🎯</span> Focus Point
                                </div>
                                <p className="text-gray-200 text-xs leading-relaxed font-medium">{feedbackJson.focus_point}</p>
                              </div>
                            )}
                          </div>
                        ) : (
                          /* Fallback to raw feedback if JSON not available */
                          <div className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
                            {feedback.feedback}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Custom Scrollbar Styles */}
      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(55, 65, 81, 0.3);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(156, 163, 175, 0.7);
        }
      `}</style>
    </div>
  );
}