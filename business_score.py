import numpy as np
from sklearn.metrics import confusion_matrix

# Coûts métier définis selon le contexte financier
COST_FN = 10  # défaut non détecté = perte du capital prêté
COST_FP = 1   # bon client refusé = manque à gagner sur intérêts

def business_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """
    Calcule le coût métier total à partir des prédictions.
    Plus le score est bas, meilleur est le modèle.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost

def business_score_normalized(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """
    Version normalisée : coût moyen par prédiction (comparable entre datasets 
    de tailles différentes). Utile pour comparer plusieurs modèles.
    """
    total_cost = business_cost(y_true, y_pred, cost_fn, cost_fp)
    return total_cost / len(y_true)

def print_cost_breakdown(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """Affiche le détail du calcul pour comprendre d'où vient le score."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Vrais négatifs (bon client, bien classé)  : {tn}")
    print(f"Faux positifs (bon client refusé)         : {fp}  -> coût: {fp * cost_fp}")
    print(f"Faux négatifs (défaut non détecté)        : {fn}  -> coût: {fn * cost_fn}")
    print(f"Vrais positifs (défaut bien détecté)      : {tp}")
    print(f"\nCoût total: {fp * cost_fp + fn * cost_fn}")
    print(f"Coût moyen par prédiction: {(fp * cost_fp + fn * cost_fn) / len(y_true):.3f}")