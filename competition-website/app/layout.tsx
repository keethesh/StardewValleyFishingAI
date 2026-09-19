import type { Metadata } from "next";
import { Inter, Press_Start_2P } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const pressStart2P = Press_Start_2P({
  weight: "400",
  variable: "--font-pixel",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Stardew Fishing AI | Video Companion",
  description: "Try the Stardew Valley fishing AI from the video, play the minigame yourself, or train a model that can beat it.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${pressStart2P.variable} antialiased bg-slate-950`}
      >
        {children}
      </body>
    </html>
  );
}
