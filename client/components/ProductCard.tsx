"use client"

import React from "react"
import Image from "next/image"
import Link from "next/link"
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Star } from "lucide-react"

export interface ProductCardData {
  title: string
  price?: string
  imageUrl?: string
  productUrl?: string
  platform?: string
  rating?: string
  description?: string
}

export const ProductCard: React.FC<ProductCardData> = ({
  title,
  price,
  imageUrl,
  productUrl,
  platform,
  rating,
  description,
}) => {
  // Helper to attempt to parse rating string like "4.5 out of 5 stars" to a number
  const parseRating = (ratingStr?: string): number | null => {
    if (!ratingStr) return null
    const match = ratingStr.match(/^[0-9.]+/);
    if (match) return parseFloat(match[0]);
    return null;
  }
  const numericRating = parseRating(rating);

  return (
    <Card className="w-full max-w-sm flex flex-col overflow-hidden h-full border-glow-primary glass-card hover-lift">
      {imageUrl && (
        <CardHeader className="p-0 relative aspect-[4/3] w-full">
          <Image
            src={imageUrl}
            alt={title || "Product image"}
            layout="fill"
            objectFit="contain" // Use "contain" to see whole image, "cover" to fill
            className="bg-white" // Add a white background for transparent images
            onError={(e) => {
              // Fallback if image fails to load, e.g., hide or show placeholder
              e.currentTarget.style.display = 'none';
            }}
          />
        </CardHeader>
      )}
      <CardContent className="p-4 flex-grow space-y-2">
        {platform && (
          <Badge variant="outline" className="text-xs mb-1 border-primary/30 text-primary">
            {platform}
          </Badge>
        )}
        <CardTitle className="text-lg font-semibold leading-tight hover:text-primary transition-colors">
          {productUrl ? (
            <Link href={productUrl} target="_blank" rel="noopener noreferrer">
              {title}
            </Link>
          ) : (
            title
          )}
        </CardTitle>
        {description && (
          <CardDescription className="text-xs text-gray-400 line-clamp-2">
            {description}
          </CardDescription>
        )}

        <div className="flex items-center justify-between mt-2">
          {price && <p className="text-xl font-bold text-primary">{price}</p>}
          {numericRating && (
            <div className="flex items-center gap-1 text-sm text-amber-400">
              <Star className="w-4 h-4 fill-amber-400" />
              <span>{numericRating.toFixed(1)}</span>
            </div>
          )}
        </div>
      </CardContent>
      {productUrl && (
        <CardFooter className="p-4 pt-0 mt-auto">
          <Button asChild variant="outline" className="w-full gradient-primary glow-primary hover-lift text-white">
            <Link href={productUrl} target="_blank" rel="noopener noreferrer">
              View Product
            </Link>
          </Button>
        </CardFooter>
      )}
    </Card>
  )
}
