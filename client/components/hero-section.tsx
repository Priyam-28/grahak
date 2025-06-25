"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Search, Sparkles, Zap, Target } from "lucide-react"

export function HeroSection() {
  return (
    <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 mb-6 glass-card px-4 py-2 rounded-full">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">AI-Powered Product Discovery</span>
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6">
            Find Products <span className="text-gradient-primary">Smarter</span>
            <br />
            Not Harder
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
            Discover products from any website using our advanced AI technology. Simply paste a URL and ask what you&#39;re
            looking for.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link href="/search">
              <Button size="lg" className="gradient-primary glow-primary hover-lift text-lg px-8 py-4">
                <Search className="w-5 h-5 mr-2" />
                Start Searching Now
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="text-lg px-8 py-4 glass-card hover-lift">
              Watch Demo
            </Button>
          </div>
        </div>

        {/* Hero Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-20">
          <Card className="glass-card hover-lift animate-float">
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 gradient-primary rounded-lg flex items-center justify-center mx-auto mb-4 glow-primary">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
              <p className="text-muted-foreground">Get results in seconds with our optimized AI algorithms</p>
            </CardContent>
          </Card>

          <Card className="glass-card hover-lift animate-float" style={{ animationDelay: "0.2s" }}>
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 gradient-secondary rounded-lg flex items-center justify-center mx-auto mb-4 glow-accent">
                <Target className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Precise Results</h3>
              <p className="text-muted-foreground">Find exactly what you#39;re looking for with intelligent matching</p>
            </CardContent>
          </Card>

          <Card className="glass-card hover-lift animate-float" style={{ animationDelay: "0.4s" }}>
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 gradient-primary rounded-lg flex items-center justify-center mx-auto mb-4 glow-primary">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">AI Powered</h3>
              <p className="text-muted-foreground">Advanced machine learning for better product discovery</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
