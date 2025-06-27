/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Mic, Square, MicOff } from "lucide-react"
import { cn } from "@/lib/utils"

interface VoiceInputProps {
  onTranscript: (transcript: string) => void
  disabled?: boolean
  className?: string
}

export function VoiceInput({ onTranscript, disabled = false, className }: VoiceInputProps) {
  const [isListening, setIsListening] = useState(false)
  const [isSupported, setIsSupported] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const recognitionRef = useRef<any | null>(null)

  useEffect(() => {
    // Check if speech recognition is supported
    if (typeof window !== "undefined") {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

      setIsSupported(!!SpeechRecognition)

      if (SpeechRecognition) {
        try {
          const recognition = new SpeechRecognition()
          recognition.continuous = false
          recognition.interimResults = false
          recognition.lang = "en-US"

          recognition.onstart = () => {
            setIsListening(true)
            setError(null)
          }

          recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript
            onTranscript(transcript)
          }

          recognition.onend = () => {
            setIsListening(false)
          }

          recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error)
            setError(event.error)
            setIsListening(false)
          }

          recognitionRef.current = recognition
        } catch (err) {
          console.error("Failed to initialize speech recognition:", err)
          setIsSupported(false)
        }
      }
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop()
        } catch (err) {
          console.error("Error stopping recognition:", err)
        }
      }
    }
  }, [onTranscript])

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      try {
        recognitionRef.current.start()
      } catch (err) {
        console.error("Error starting recognition:", err)
        setError("Failed to start voice recognition")
      }
    }
  }

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      try {
        recognitionRef.current.stop()
      } catch (err) {
        console.error("Error stopping recognition:", err)
      }
    }
  }

  if (!isSupported) {
    return (
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled
        className={cn("opacity-50", className)}
        title="Voice input not supported in this browser"
      >
        <MicOff className="w-4 h-4" />
      </Button>
    )
  }

  return (
    <Button
      type="button"
      variant={isListening ? "destructive" : "outline"}
      size="icon"
      onClick={isListening ? stopListening : startListening}
      disabled={disabled}
      className={cn(
        "transition-all duration-200",
        isListening && "animate-pulse-glow",
        error && "border-destructive",
        className,
      )}
      title={error ? `Voice error: ${error}` : isListening ? "Click to stop recording" : "Click to start voice input"}
    >
      {isListening ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
    </Button>
  )
}
