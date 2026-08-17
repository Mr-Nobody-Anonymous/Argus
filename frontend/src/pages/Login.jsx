/**
 * Login screen.
 *
 * The API requires a bearer token on every route except /health and the auth
 * endpoints themselves, so the dashboard cannot render anything useful until a
 * token exists. This is the gate.
 *
 * Credentials are verified against the Django `auth_user` table by the backend -
 * Argus deliberately has no second user store to drift out of sync.
 */
import React, { useState } from 'react';
import {
    Box,
    Paper,
    TextField,
    Button,
    Typography,
    Alert,
    CircularProgress,
    Stack,
} from '@mui/material';
import { Shield } from '@mui/icons-material';

import { authAPI } from '../services/api';

export default function Login({ onSuccess }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);

    const submit = async (event) => {
        event.preventDefault();
        setBusy(true);
        setError(null);
        try {
            const data = await authAPI.login(username, password);
            onSuccess?.(data.user);
        } catch (err) {
            const status = err.response?.status;
            if (status === 429) {
                // The backend locks an account after 5 failed attempts. Say so
                // plainly rather than letting the operator keep guessing.
                setError(
                    err.response?.data?.detail ||
                    'Too many failed attempts. This account is temporarily locked.'
                );
            } else if (status === 401) {
                setError('Incorrect username or password.');
            } else {
                setError(err.response?.data?.detail || 'Unable to reach the Argus API.');
            }
        } finally {
            setBusy(false);
        }
    };

    return (
        <Box
            sx={{
                minHeight: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'background.default',
                p: 2,
            }}
        >
            <Paper
                elevation={6}
                sx={{ p: 4, width: '100%', maxWidth: 400, border: '1px solid #1a1a1a' }}
            >
                <Stack spacing={1} alignItems="center" sx={{ mb: 3 }}>
                    <Shield sx={{ fontSize: 48, color: 'primary.main' }} />
                    <Typography variant="h5" fontWeight={600}>
                        Argus
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Sign in to the monitoring console
                    </Typography>
                </Stack>

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                <form onSubmit={submit}>
                    <Stack spacing={2}>
                        <TextField
                            label="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            autoFocus
                            fullWidth
                            autoComplete="username"
                            disabled={busy}
                        />
                        <TextField
                            label="Password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            fullWidth
                            autoComplete="current-password"
                            disabled={busy}
                        />
                        <Button
                            type="submit"
                            variant="contained"
                            size="large"
                            disabled={busy || !username || !password}
                            startIcon={busy ? <CircularProgress size={18} /> : null}
                        >
                            {busy ? 'Signing in…' : 'Sign in'}
                        </Button>
                    </Stack>
                </form>

                <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: 'block', mt: 3, textAlign: 'center' }}
                >
                    Roles: viewer (read-only) · operator (zones, tracking) · admin (full)
                </Typography>
            </Paper>
        </Box>
    );
}
