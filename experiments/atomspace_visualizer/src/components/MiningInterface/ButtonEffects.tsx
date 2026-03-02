import { createSignal, onCleanup, onMount } from 'solid-js';

interface ParticleProps {
  active: boolean;
}

export const ButtonParticleEffects = (props: ParticleProps) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let animationFrame: number;
  let particlesArray: Array<{
    x: number;
    y: number;
    size: number;
    speedX: number;
    speedY: number;
    color: string;
    opacity: number;
  }> = [];

  const [dimensions, setDimensions] = createSignal({ width: 300, height: 150 });
  
  const colors = ['#ff9500', '#ff00cc', '#ffeb3b', '#ffffff'];
  
  const createParticles = (x: number, y: number, count: number) => {
    for (let i = 0; i < count; i++) {
      particlesArray.push({
        x,
        y,
        size: Math.random() * 5 + 1,
        speedX: Math.random() * 3 - 1.5,
        speedY: Math.random() * -3 - 1,
        color: colors[Math.floor(Math.random() * colors.length)],
        opacity: 1
      });
    }
  };

  const animate = () => {
    if (!canvasRef) return;
    
    const ctx = canvasRef.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, dimensions().width, dimensions().height);
    
    // Update particles
    for (let i = 0; i < particlesArray.length; i++) {
      const p = particlesArray[i];
      
      p.x += p.speedX;
      p.y += p.speedY;
      p.size -= 0.05;
      p.opacity -= 0.01;
      
      ctx.globalAlpha = p.opacity > 0 ? p.opacity : 0;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Remove small or invisible particles
    particlesArray = particlesArray.filter(p => p.size > 0.2 && p.opacity > 0);
    
    // Create new particles if active
    if (props.active && Math.random() > 0.8) {
      const buttonRect = canvasRef.getBoundingClientRect();
      const x = Math.random() * buttonRect.width;
      const y = buttonRect.height - 5;
      createParticles(x, y, 1);
    }
    
    animationFrame = requestAnimationFrame(animate);
  };
  
  onMount(() => {
    if (!canvasRef) return;
    
    const resizeCanvas = () => {
      const parent = canvasRef?.parentElement;
      if (parent) {
        setDimensions({
          width: parent.clientWidth,
          height: parent.clientHeight
        });
      }
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    animationFrame = requestAnimationFrame(animate);
    
    onCleanup(() => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrame);
    });
  });

  return (
    <canvas
      ref={canvasRef}
      width={dimensions().width}
      height={dimensions().height}
      style={{
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        'pointer-events': 'none',
        'z-index': '0'
      }}
    />
  );
};

export default ButtonParticleEffects;
