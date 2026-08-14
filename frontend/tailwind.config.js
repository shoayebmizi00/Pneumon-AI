/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#102a2a', mist: '#f6faf9', teal: {50:'#edf9f7',100:'#d8f1ec',200:'#b4e3da',500:'#16877a',600:'#0f7167',700:'#0e5b55',900:'#133c3a'},
        amber: {50:'#fff9eb',200:'#f7df9b',600:'#a36108'}, rose: {50:'#fff1f2',200:'#fecdd3',700:'#be123c'},
      },
      fontFamily: { sans: ['Figtree', 'Noto Sans', 'Segoe UI', 'sans-serif'] },
      boxShadow: { soft: '0 20px 60px -35px rgba(15, 67, 63, .35)', card: '0 12px 32px -24px rgba(15, 67, 63, .3)' },
    },
  },
  plugins: [],
}
