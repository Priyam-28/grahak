"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Search, Globe, MessageCircle, Sparkles, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface Message {
  role: "user" | "assistant"
  content: string
}

export default function ProductFinder() {
  const [url, setUrl] = useState("")
  const [query, setQuery] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState<"url" | "chat">("url")
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) return

    setCurrentStep("chat")
    setMessages([
      {
        role: "assistant",
        content: `Great! I'm ready to help you find products from ${url}. What are you looking for?`,
      },
    ])
    setError(null)
  }

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userMessage: Message = { role: "user", content: query }
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)
    setQuery("")
    setError(null)

    // Cancel previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch("http://localhost:8000/api/v1/findproduct/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          url,
          query: query,
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

  const resetChat = () => {
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    setCurrentStep("url")
    setMessages([])
    setUrl("")
    setQuery("")
    setError(null)
    setLoading(false)
  }

  // Cleanup on unmount
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
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-3 rounded-full bg-gradient-to-r from-orange-500 to-red-500 shadow-lg shadow-orange-500/25 glow-orange">
              <Search className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gradient-orange">AI Product Finder</h1>
          </div>
          <p className="text-gray-400 text-lg">Discover products from any website using AI-powered search</p>
        </div>

        {/* Error Display */}
        {error && (
          <Card className="mb-6 bg-red-900/20 border-red-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {currentStep === "url" ? (
          /* URL Input Step */
          <Card className="backdrop-orange border-glow-orange">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Globe className="w-5 h-5 text-orange-400" />
                Enter Website URL
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleUrlSubmit} className="space-y-4">
                <div className="relative">
                  <Input
                    type="url"
                    placeholder="https://example-store.com"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="border-glow-orange bg-black/50 text-white placeholder-gray-500"
                    required
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white glow-orange"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  Start Product Search
                </Button>
              </form>
            </CardContent>
          </Card>
        ) : (
          /* Chat Interface */
          <div className="space-y-6">
            {/* Website Badge */}
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="border-orange-500/30 text-orange-400 bg-orange-500/10">
                <Globe className="w-3 h-3 mr-1" />
                {new URL(url).hostname}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={resetChat}
                className="border-orange-500/30 text-orange-400 hover:bg-orange-500/10"
                disabled={loading}
              >
                Change Website
              </Button>
            </div>

            {/* Chat Messages */}
            <Card className="backdrop-orange border-glow-orange">
              <CardContent className="p-6">
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={cn("flex gap-3", message.role === "user" ? "justify-end" : "justify-start")}
                    >
                      <div
                        className={cn(
                          "max-w-[80%] p-3 rounded-lg",
                          message.role === "user"
                            ? "bg-gradient-to-r from-orange-500 to-red-500 text-white glow-orange"
                            : "bg-gray-800/50 text-gray-100 border border-gray-700/50",
                        )}
                      >
                        <div className="flex items-start gap-2">
                          {message.role === "assistant" && (
                            <MessageCircle className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                          )}
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="bg-gray-800/50 text-gray-100 border border-gray-700/50 p-3 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-4 h-4 text-orange-400 animate-spin" />
                          <span className="text-sm">Searching for products...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Query Input */}
            <Card className="backdrop-orange border-glow-orange">
              <CardContent className="p-4">
                <form onSubmit={handleQuerySubmit} className="flex gap-2">
                  <Input
                    placeholder="What products are you looking for?"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="border-glow-orange bg-black/50 text-white placeholder-gray-500"
                    disabled={loading}
                  />
                  <Button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white glow-orange"
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
