import React, { useState } from 'react';
import { Login } from './Login';
import { Signup } from './Signup';
import { EmailConfirmation } from './EmailConfirmation';

interface AuthProps {
  onLoginSuccess: () => void;
}

type AuthState = 'login' | 'signup' | 'confirmation';

export const Auth: React.FC<AuthProps> = ({ onLoginSuccess }) => {
  const [authState, setAuthState] = useState<AuthState>('login');
  const [email, setEmail] = useState('');

  const handleSignupSuccess = (email: string) => {
    setEmail(email);
    setAuthState('confirmation');
  };

  const handleConfirmationSuccess = () => {
    // After confirmation, we can either auto-login or ask them to login
    // For simplicity, let's ask them to login
    setAuthState('login');
  };

  return (
   <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-background via-muted/30 to-background p-4 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-secondary/10 rounded-full blur-3xl translate-x-1/2 translate-y-1/2" />
        
            <div className="w-full max-w-md relative z-10 animate-fade-in">
                <div className="text-center mb-8">
                <h1 className="text-4xl font-bold bg-linear-to-r from-primary via-secondary to-primary bg-clip-text text-transparent mb-2">
                Lumina
                </h1>
                <p className="text-muted-foreground text-sm">
                AI-Powered Skin Lesion Detection
                </p>
            </div>

            {authState === 'login' && (
                <Login
                onSuccess={onLoginSuccess}
                onSwitchToSignup={() => setAuthState('signup')}
                />
            )}

            {authState === 'signup' && (
                <Signup
                onSuccess={handleSignupSuccess}
                onSwitchToLogin={() => setAuthState('login')}
                />
            )}

            {authState === 'confirmation' && (
                <EmailConfirmation
                email={email}
                onSuccess={handleConfirmationSuccess}
                onCancel={() => setAuthState('login')}
                />
            )}
        </div>
    </div>
  );
};
