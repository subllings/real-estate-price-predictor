import React from "react";
import styles from "./ResultCard.module.css";

const ResultCard = ({ title, value, className }) => {
  const formattedValue = value
    .toFixed(0)
    .replace(/\B(?=(\d{3})+(?!\d))/g, " ");

  // Découper le titre avant et après les mots-clés pour ne mettre en gras que la fin
  const [prefix, boldPart] = title.split(/(all features|top 30 features)/i);

  return (
    <div className={`${styles.resultCard} ${className || ''}`}>
      <h3 className={styles.title}>
        {prefix}
        <span className={styles.boldTitle}>{boldPart}</span>
      </h3>
      <div className={styles.price}>
        <span>Estimated Price (€):</span>
        <span className={styles.priceValue}>{formattedValue}</span>
      </div>
    </div>
  );
};

export default ResultCard;
