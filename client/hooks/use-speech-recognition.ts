/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"

import { useState, useRef, useEffect, useCallback } from "react"

interface UseSpeechRecognitionOptions {
  continuous?: boolean
  interimResults?: boolean
  lang?: string
}

interface UseSpeechRecognitionReturn {
  isListening: boolean
  isSupported: boolean
  transcript: string
  error: string | null
  startListening: () => void
  stopListening: () => void
  resetTranscript: () => void
}

export function useSpeechRecognition(options: UseSpeechRecognitionOptions = {}): UseSpeechRecognitionReturn {
  const { continuous = false, interimResults = false, lang = "en-US" } = options

  const [isListening, setIsListening] = useState(false)
  const [isSupported, setIsSupported] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [error, setError] = useState<string | null>(null)
  const recognitionRef = useRef<any | null>(null)

  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

      if (SpeechRecognition) {
        setIsSupported(true)

        try {
          const recognition = new SpeechRecognition()
          recognition.continuous = continuous
          recognition.interimResults = interimResults
          recognition.lang = lang

          recognition.onstart = () => {
            setIsListening(true)
            setError(null)
          }

          recognition.onresult = (event) => {
            let finalTranscript = ""
            let interimTranscript = ""

            for (let i = event.resultIndex; i < event.results.length; i++) {
              const result = event.results[i]
              if (result.isFinal) {
                finalTranscript += result[0].transcript
              } else {
                interimTranscript += result[0].transcript
              }
            }

            setTranscript(finalTranscript || interimTranscript)
          }

          recognition.onend = () => {
            setIsListening(false)
          }

          recognition.onerror = (event) => {
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
  }, [continuous, interimResults, lang])

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      try {
        setTranscript("")
        setError(null)
        recognitionRef.current.start()
      } catch (err) {
        console.error("Error starting recognition:", err)
        setError("Failed to start voice recognition")
      }
    }
  }, [isListening])

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      try {
        recognitionRef.current.stop()
      } catch (err) {
        console.error("Error stopping recognition:", err)
      }
    }
  }, [isListening])

  const resetTranscript = useCallback(() => {
    setTranscript("")
    setError(null)
  }, [])

  return {
    isListening,
    isSupported,
    transcript,
    error,
    startListening,
    stopListening,
    resetTranscript,
  }
}
