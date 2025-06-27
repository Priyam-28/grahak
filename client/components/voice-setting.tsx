"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Volume2, Settings } from "lucide-react"

interface Voice {
  id: string
  name: string
  language: string
}

interface VoiceSettingsProps {
  voiceEnabled: boolean
  onVoiceEnabledChange: (enabled: boolean) => void
  selectedVoice: string
  onVoiceChange: (voiceId: string) => void
  className?: string
}

export function VoiceSettings({
  voiceEnabled,
  onVoiceEnabledChange,
  selectedVoice,
  onVoiceChange,
  className,
}: VoiceSettingsProps) {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchVoices()
  }, [])

  const fetchVoices = async () => {
    setLoading(true)
    try {
      const response = await fetch("http://localhost:8000/api/v1/findproduct/voices")
      const data = await response.json()
      if (data.success) {
        setVoices(data.voices)
      }
    } catch (error) {
      console.error("Failed to fetch voices:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className={`glass-card ${className}`}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Settings className="w-4 h-4" />
          Voice Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="voice-enabled" className="text-sm">
            Enable Text-to-Speech
          </Label>
          <Switch id="voice-enabled" checked={voiceEnabled} onCheckedChange={onVoiceEnabledChange} />
        </div>

        {voiceEnabled && (
          <div className="space-y-2">
            <Label className="text-sm">Voice Selection</Label>
            <Select value={selectedVoice} onValueChange={onVoiceChange} disabled={loading}>
              <SelectTrigger className="glass-card">
                <SelectValue placeholder="Select a voice" />
              </SelectTrigger>
              <SelectContent>
                {voices.map((voice) => (
                  <SelectItem key={voice.id} value={voice.id}>
                    <div className="flex items-center gap-2">
                      <Volume2 className="w-3 h-3" />
                      {voice.name}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
