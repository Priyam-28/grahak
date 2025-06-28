/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Search, MessageCircle, Sparkles, AlertCircle, ArrowLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { VoiceInput } from "@/components/voice-input"

import { VoiceSettings } from "@/components/voice-setting"
import { AudioPlayer } from "./audio-player"

interface Message {
  role: "user" | "assistant"
  content: string
  audio?: string
}

export function SearchInterface() {
  const [searchQuery, setSearchQuery] = useState("")
  const [query, setQuery] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState<"search" | "chat">("search")
  const [error, setError] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [selectedVoice, setSelectedVoice] = useState("en-US-terrell")
  const [showVoiceSettings, setShowVoiceSettings] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    setCurrentStep("chat")
    setMessages([
      {
        role: "assistant",
        content: `Perfect! I'm ready to help you find "${searchQuery}" products. Ask me anything about these products or related items!`,
      },
    ])
    setError(null)
  }

  const handleVoiceTranscript = (transcript: string) => {
    setQuery(transcript)
  }

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userMessage: Message = { role: "user", content: query }
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)
    setQuery("")
    setError(null)

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch("http://localhost:8000/api/v1/findproduct/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          query: searchQuery + " " + query,
          voice_enabled: voiceEnabled,
          voice_id: selectedVoice,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.success) {
        const assistantMessage: Message = {
          role: "assistant",
          content: data.response || "No products found matching your criteria.",
          audio: data.audio || undefined,
        }
        setMessages((prev) => [...prev, assistantMessage])
      } else {
        throw new Error(data.message || "Failed to find products")
      }
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Request was aborted")
        return
      }

      const errorMessage = error.message || "An unexpected error occurred"
      setError(errorMessage)

      const errorResponse: Message = {
        role: "assistant",
        content: "Sorry, I encountered an error while searching for products. Please try again.",
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setLoading(false)
      abortControllerRef.current = null
    }
  }

  const resetSearch = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    setCurrentStep("search")
    setMessages([])
    setSearchQuery("")
    setQuery("")
    setError(null)
    setLoading(false)
  }

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  return (
    <div className="min-h-screen p-4 flex items-center justify-center">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 mb-4 text-muted-foreground hover:text-primary transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>

          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-3 rounded-full gradient-primary glow-primary animate-pulse-glow">
              <Search className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gradient-primary">AI Product Search</h1>
          </div>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Search and discover products using intelligent AI search with voice support
          </p>
        </div>

        {/* Voice Settings */}
        {currentStep === "chat" && (
          <div className="mb-6">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowVoiceSettings(!showVoiceSettings)}
              className="glass-card mb-4"
            >
              Voice Settings
            </Button>
            {showVoiceSettings && (
              <VoiceSettings
                voiceEnabled={voiceEnabled}
                onVoiceEnabledChange={setVoiceEnabled}
                selectedVoice={selectedVoice}
                onVoiceChange={setSelectedVoice}
                className="mb-4"
              />
            )}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <Card className="mb-6 glass-card border-destructive/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {currentStep === "search" ? (
          /* Search Input Step */
          <Card className="glass-card glow-primary hover-lift">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5 text-primary" />
                Search for Products
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSearchSubmit} className="space-y-6">
                <div className="relative">
                  <Input
                    type="text"
                    placeholder="e.g., wireless headphones, laptop, smartphone"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="glass-card text-lg py-6 px-4 border-primary/30 focus:border-primary focus:ring-primary/20"
                    required
                  />
                </div>
                <Button
                  type="submit"
                  size="lg"
                  className="w-full gradient-primary glow-primary hover-lift text-lg py-6"
                >
                  <Sparkles className="w-5 h-5 mr-2" />
                  Start Product Search
                </Button>
              </form>
            </CardContent>
          </Card>
        ) : (
          /* Chat Interface */
          <div className="space-y-6">
            {/* Search Query Badge */}
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="glass-card border-primary/30 text-primary px-4 py-2">
                <Search className="w-4 h-4 mr-2" />
                Searching for: {searchQuery}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={resetSearch}
                className="glass-card hover-lift"
                disabled={loading}
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                New Search
              </Button>
            </div>

            {/* Chat Messages */}
            <Card className="glass-card glow-primary">
              <CardContent className="p-6">
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={cn("flex gap-3", message.role === "user" ? "justify-end" : "justify-start")}
                    >
                      <div
                        className={cn(
                          "max-w-[80%] p-4 rounded-xl",
                          message.role === "user"
                            ? "gradient-primary text-white glow-primary"
                            : "glass-card border-primary/20",
                        )}
                      >
                        <div className="flex items-start gap-2">
                          {message.role === "assistant" && (
                            <MessageCircle className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                          )}
                          <div className="flex-1">
                            <p className="text-sm leading-relaxed whitespace-pre-wrap mb-2">{message.content}</p>
                            {message.audio && <AudioPlayer audioBase64={message.audio} autoPlay={voiceEnabled} />}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="glass-card border-primary/20 p-4 rounded-xl">
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-4 h-4 text-primary animate-spin" />
                          <span className="text-sm">
                            {voiceEnabled ? "Searching and generating audio..." : "Searching for products..."}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Query Input */}
            <Card className="glass-card">
              <CardContent className="p-4">
                <form onSubmit={handleQuerySubmit} className="flex gap-3">
                  <Input
                    placeholder="Ask me about these products or search for more..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="glass-card border-primary/30 focus:border-primary focus:ring-primary/20"
                    disabled={loading}
                  />
                  <VoiceInput
                    onTranscript={handleVoiceTranscript}
                    disabled={loading}
                    className="glass-card border-primary/30"
                  />
                  <Button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="gradient-primary glow-primary hover-lift px-6"
                  >
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
