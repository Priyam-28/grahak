import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { CTASection } from "@/components/cta-section"
import { Navbar } from "@/components/navbar"

export default function Home() {
  return (
    <main className="min-h-screen gradient-hero">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <CTASection />
    </main>
  )
}
