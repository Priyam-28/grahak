/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"

import React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Search, MessageCircle, AlertCircle, ArrowLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { VoiceInput } from "@/components/voice-input"

import { VoiceSettings } from "@/components/voice-setting"
import { AudioPlayer } from "./audio-player"
import { ProductCard, type ProductCardData } from "./ProductCard" // Import ProductCard

interface Message {
  role: "user" | "assistant"
  content: string // For LLM text summary
  audio?: string
  products?: ProductCardData[] // For structured product list
}

export function SearchInterface() {
  // const [searchQuery, setSearchQuery] = useState("") // Removed searchQuery
  const [query, setQuery] = useState("") // This will be the main search query
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  // const [currentStep, setCurrentStep] = useState<"search" | "chat">("search") // Removed currentStep, always in "chat" mode essentially
  const [error, setError] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [selectedVoice, setSelectedVoice] = useState("en-US-terrell")
  const [showVoiceSettings, setShowVoiceSettings] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [currentSearchTopic, setCurrentSearchTopic] = useState<string>("") // To store the topic of the current search

  // New state for super search
  const [selectedSites, setSelectedSites] = useState<string[]>(() => {
    if (typeof window !== "undefined") {
      const savedSites = localStorage.getItem("selectedSites")
      return savedSites ? JSON.parse(savedSites) : ["amazon"] // Default to Amazon
    }
    return ["amazon"]
  })
  const [minPrice, setMinPrice] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("minPrice") || ""
    }
    return ""
  })
  const [maxPrice, setMaxPrice] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("maxPrice") || ""
    }
    return ""
  })


  // Removed handleSearchSubmit as the initial search step is gone

  const handleVoiceTranscript = (transcript: string) => {
    setQuery(transcript) // Voice input directly sets the main query
  }

  // Effect for loading from localStorage on mount
  useEffect(() => {
    const savedQuery = localStorage.getItem("searchQuery")
    if (savedQuery) setQuery(savedQuery)

    const savedSites = localStorage.getItem("selectedSites")
    if (savedSites) setSelectedSites(JSON.parse(savedSites))

    const savedMinPrice = localStorage.getItem("minPrice")
    if (savedMinPrice) setMinPrice(savedMinPrice)

    const savedMaxPrice = localStorage.getItem("maxPrice")
    if (savedMaxPrice) setMaxPrice(savedMaxPrice)
  }, [])

  // Effect for saving to localStorage when values change
  useEffect(() => {
    localStorage.setItem("searchQuery", query)
  }, [query])

  useEffect(() => {
    localStorage.setItem("selectedSites", JSON.stringify(selectedSites))
  }, [selectedSites])

  useEffect(() => {
    localStorage.setItem("minPrice", minPrice)
  }, [minPrice])

  useEffect(() => {
    localStorage.setItem("maxPrice", maxPrice)
  }, [maxPrice])


  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userMessage: Message = { role: "user", content: query }
    // If it's a new search topic (no messages yet, or last message was assistant), clear old messages.
    const newSearchTopic = messages.length === 0 || messages[messages.length -1].role === "assistant"

    if (newSearchTopic) {
      setMessages([userMessage])
      setCurrentSearchTopic(query) // Set the new search topic
    } else {
      setMessages((prev) => [...prev, userMessage])
    }

    setLoading(true)
    const currentQueryForRequest = query; // store current query for the request
    setQuery("") // Clear input field after submit
    setError(null)

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    // Construct query parameters for the new API
    const apiParams = new URLSearchParams({
      query: currentQueryForRequest,
    });
    selectedSites.forEach(site => apiParams.append("sites", site));
    if (minPrice) apiParams.append("min_price", minPrice);
    if (maxPrice) apiParams.append("max_price", maxPrice);
    // apiParams.append("max_results_per_site", "3"); // Defaulting to 3, or make it configurable

    try {
      // const requestBody = { // Not needed for GET usually, but if API changes to POST
      //   query: currentQueryForRequest,
      //   sites: selectedSites,
      //   min_price: minPrice ? parseFloat(minPrice) : undefined,
      //   max_price: maxPrice ? parseFloat(maxPrice) : undefined,
      //   max_results_per_site: 3 // Or make this configurable
      // }

      const response = await fetch(`http://localhost:8000/api/v1/super-search/products?${apiParams.toString()}`, {
        method: "GET", // Changed to GET as per new API design for queries
        headers: {
          "Content-Type": "application/json", // Still good practice, though GET has no body
        },
        // body: JSON.stringify(requestBody), // No body for GET
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        // Try to parse error from FastAPI
        const errorData = await response.json().catch(() => null);
        const detail = errorData?.detail || `HTTP error! status: ${response.status}`;
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }

      const productsData: ProductCardData[] = await response.json()

      // Create a simple assistant message.
      // The new API directly returns product list or error. No separate llm_summary or audio.
      let assistantContent = "Here are the products I found based on your criteria:"
      if (!productsData || productsData.length === 0) {
        assistantContent = "I couldn't find any products matching your criteria. Please try adjusting your search."
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: assistantContent,
        products: productsData || [],
        // audio: undefined, // No audio from this endpoint
      }
      setMessages((prev) => [...prev, assistantMessage])

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

    // setCurrentStep("search") // Removed, no longer switching steps
    setMessages([])
    // setSearchQuery("") // Removed
    setQuery("")
    setCurrentSearchTopic("") // Reset current search topic
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

        {/* Voice Settings - always show if messages exist (i.e., a search has been made) */}
        {messages.length > 0 && (
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

        {/* Always show chat interface style */}
        <div className="space-y-6">
          {/* Conditional Search Topic Display and New Search Button */}
          {messages.length > 0 && currentSearchTopic && (
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="glass-card border-primary/30 text-primary px-4 py-2">
                <Search className="w-4 h-4 mr-2" />
                Topic: {currentSearchTopic}
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
          )}

          {/* Chat Messages Area: Only show if there are messages */}
          {messages.length > 0 && (
            <Card className="glass-card glow-primary">
              <CardContent className="p-6">
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {messages.map((message, index) => (
                    <React.Fragment key={index}>
                      <div
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
                          <div className="flex-1">
                             <p className="text-sm leading-relaxed whitespace-pre-wrap mb-2">{message.content}</p>
                            {message.audio && <AudioPlayer audioBase64={message.audio} autoPlay={voiceEnabled} />}
                          </div>
                        </div>
                      </div>
                    {/* Render Product Cards if they exist for an assistant message */}
                    {message.role === "assistant" && message.products && message.products.length > 0 && (
                      <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 px-2 py-2 -mx-2 max-h-96 overflow-y-auto">
                        {message.products.map((product, pIndex) => (
                          <ProductCard key={`${index}-${pIndex}`} {...product} />
                        ))}
                      </div>
                    )}
                    </React.Fragment>
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
          )}

          {/* Query Input - This is now the main search input */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base"> {/* Adjusted title size */}
                <Search className="w-5 h-5 text-primary" />
                What product are you looking for?
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-4">
              <form onSubmit={handleQuerySubmit} className="space-y-4">
                <div className="flex gap-3">
                  <Input
                    placeholder="e.g., best wireless headphones under $100, red running shoes size 10"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="flex-grow glass-card border-primary/30 focus:border-primary focus:ring-primary/20 text-base py-3 px-4"
                    disabled={loading}
                    required
                  />
                  <VoiceInput
                    onTranscript={handleVoiceTranscript}
                    disabled={loading}
                    className="glass-card border-primary/30"
                  />
                </div>

                {/* Site Selection Checkboxes */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Select Sites:</label>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {["amazon", "flipkart", "myntra"].map((site) => (
                      <label key={site} className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          className="form-checkbox h-5 w-5 text-primary-600 border-gray-300 rounded focus:ring-primary-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-offset-gray-800"
                          checked={selectedSites.includes(site)}
                          onChange={() => {
                            setSelectedSites(prev =>
                              prev.includes(site) ? prev.filter(s => s !== site) : [...prev, site]
                            )
                          }}
                          disabled={loading}
                        />
                        <span className="text-sm text-gray-700 dark:text-gray-200 capitalize">{site}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Price Range Inputs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label htmlFor="minPrice" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Min Price (INR):</label>
                    <Input
                      id="minPrice"
                      type="number"
                      placeholder="e.g., 1000"
                      value={minPrice}
                      onChange={(e) => setMinPrice(e.target.value)}
                      className="glass-card border-primary/30 focus:border-primary focus:ring-primary/20"
                      disabled={loading}
                    />
                  </div>
                  <div className="space-y-1">
                    <label htmlFor="maxPrice" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Max Price (INR):</label>
                    <Input
                      id="maxPrice"
                      type="number"
                      placeholder="e.g., 5000"
                      value={maxPrice}
                      onChange={(e) => setMaxPrice(e.target.value)}
                      className="glass-card border-primary/30 focus:border-primary focus:ring-primary/20"
                      disabled={loading}
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  disabled={loading || !query.trim() || selectedSites.length === 0}
                  className="w-full gradient-primary glow-primary hover-lift px-6 py-3 text-base"
                  size="lg"
                >
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5 mr-2" />}
                  {loading ? "Searching..." : "Find Products"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}