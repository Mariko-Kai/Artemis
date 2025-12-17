import React, { useState, useRef } from 'react';
import { Mic, Square } from 'lucide-react';
import clsx from 'clsx';

const AudioRecorder = ({ onAudioRecorded, disabled }) => {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                onAudioRecorded(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Mic access denied", err);
            alert("Could not access microphone");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
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
