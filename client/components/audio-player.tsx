"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Play, Pause, Volume2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface AudioPlayerProps {
  audioBase64: string
  autoPlay?: boolean
  className?: string
}

export function AudioPlayer({ audioBase64, autoPlay = false, className }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (audioBase64) {
      const audioBlob = new Blob([Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0))], {
        type: "audio/mpeg",
      })
      const audioUrl = URL.createObjectURL(audioBlob)

      if (audioRef.current) {
        audioRef.current.src = audioUrl
        if (autoPlay) {
          audioRef.current.play()
        }
      }

      return () => {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioBase64, autoPlay])

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
    }
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }

  return (
    <div className={cn("flex items-center gap-2 glass-card p-2 rounded-lg", className)}>
      <audio
        ref={audioRef}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      <Button variant="ghost" size="icon" onClick={togglePlayPause} className="h-8 w-8">
        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
      </Button>

      <Volume2 className="w-4 h-4 text-muted-foreground" />

      <div className="flex-1 min-w-0">
        <div className="w-full bg-muted rounded-full h-1">
          <div
            className="bg-primary h-1 rounded-full transition-all duration-100"
            style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>
      </div>

      <span className="text-xs text-muted-foreground tabular-nums">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>
  )
}
