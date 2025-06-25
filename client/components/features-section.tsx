"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Globe, MessageSquare, Filter, Clock, Shield, Cpu } from "lucide-react"

const features = [
  {
    icon: Globe,
    title: "Universal Website Support",
    description: "Works with any e-commerce website or product catalog",
    badge: "Popular",
  },
  {
    icon: MessageSquare,
    title: "Natural Language Search",
    description: "Ask questions in plain English and get relevant results",
    badge: "New",
  },
  {
    icon: Filter,
    title: "Smart Filtering",
    description: "Automatically filters and categorizes products for you",
    badge: null,
  },
  {
    icon: Clock,
    title: "Real-time Results",
    description: "Get instant results as you type your queries",
    badge: null,
  },
  {
    icon: Shield,
    title: "Privacy First",
    description: "Your searches are private and secure",
    badge: null,
  },
  {
    icon: Cpu,
    title: "AI-Powered",
    description: "Advanced algorithms understand context and intent",
    badge: "Featured",
  },
]

export function FeaturesSection() {
  return (
    <section id="features" className="py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">
            Powerful Features for <span className="text-gradient-secondary">Smart Shopping</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our AI-powered platform makes product discovery effortless and efficient
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card key={index} className="glass-card hover-lift group">
              <CardHeader>
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 gradient-primary rounded-lg flex items-center justify-center glow-primary group-hover:animate-pulse-glow">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  {feature.badge && (
                    <Badge variant="secondary" className="gradient-secondary text-white">
                      {feature.badge}
                    </Badge>
                  )}
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
