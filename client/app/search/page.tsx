import { SearchInterface } from "@/components/search-interface"
import { Navbar } from "@/components/navbar"

export default function SearchPage() {
  return (
    <main className="min-h-screen gradient-hero">
      <Navbar />
      <div className="pt-20">
        <SearchInterface />
      </div>
    </main>
  )
}
