"use client"

import { Card, CardContent } from "@/components/ui/card"
import { MicOff } from "lucide-react"

interface VoiceFallbackProps {
  message?: string
}

export function VoiceFallback({ message = "Voice input is not supported in this browser" }: VoiceFallbackProps) {
  return (
    <Card className="glass-card border-muted/30">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <MicOff className="w-4 h-4" />
          <span className="text-sm">{message}</span>
        </div>
        <p className="text-xs text-muted-foreground mt-2">Try using Chrome, Edge, or Safari for voice input support.</p>
      </CardContent>
    </Card>
  )
}
