import React, { useState, useRef } from 'react';
import { Mic, Square } from 'lucide-react';
import clsx from 'clsx';

const AudioRecorder = ({ onAudioRecorded, disabled }) => {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const audioContextRef = useRef(null);
    const silenceTimerRef = useRef(null);
    const animationFrameRef = useRef(null);

    const cleanupAudio = () => {
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (mediaRecorderRef.current) {
            // Stop all tracks
            if (mediaRecorderRef.current.stream) {
                mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            }
        }
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            chunksRef.current = [];

            // Audio Analysis for Silence Detection
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            const analyser = audioContextRef.current.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            let lastSoundTime = Date.now();
            const silenceThreshold = 10; // Values are 0-255. 10 is very quiet background noise.
            const silenceDuration = 2000; // 2 seconds

            const updateVolume = () => {
                // Check actual recorder state, as isRecording state var might be stale in this closure
                if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') return;

                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((acc, val) => acc + val, 0) / bufferLength;

                if (average > silenceThreshold) {
                    lastSoundTime = Date.now();
                } else {
                    if (Date.now() - lastSoundTime > silenceDuration) {
                        console.log("Silence detected, stopping recording...");
                        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                            mediaRecorderRef.current.stop();
                            setIsRecording(false);
                        }
                        return;
                    }
                }
                animationFrameRef.current = requestAnimationFrame(updateVolume);
            };

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                cleanupAudio();
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                onAudioRecorded(blob);
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            updateVolume(); // Start monitoring

        } catch (err) {
            console.error("Mic access denied", err);
            alert(`Could not access microphone: ${err.message}`);
            cleanupAudio();
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            // Cleanup happens in onstop event
        }
    };

    return (
        <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={disabled}
            className={clsx(
                "p-2 rounded-full transition-all",
                isRecording
                    ? "bg-red-500 text-white animate-pulse shadow-lg ring-4 ring-red-100"
                    : "hover:bg-gray-200 text-gray-600",
                disabled && "opacity-50 cursor-not-allowed"
            )}
            title="Voice Input"
        >
            {isRecording ? <Square size={20} fill="currentColor" /> : <Mic size={20} />}
        </button>
    );
};

export default AudioRecorder;
