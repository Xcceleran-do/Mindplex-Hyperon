/** Tailwind CSS configuration (Step 10 styling). */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        brand: {
          50: '#f5f9ff',
          100: '#e0edff',
          200: '#b8d5ff',
          300: '#84b9ff',
          400: '#4d9eff',
          500: '#1e82ff',
          600: '#0066e6',
          700: '#004fb3',
          800: '#003880',
          900: '#00224d',
        },
      },
    },
  },
  plugins: [],
}
