-- MySQL dump 10.13  Distrib 8.0.36, for Linux (x86_64)
--
-- Host: 127.0.0.1    Database: Dalmia
-- ------------------------------------------------------
-- Server version	8.0.44

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Cameras`
--

DROP TABLE IF EXISTS `Cameras`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Cameras` (
  `CameraId` varchar(100) NOT NULL,
  `WorkStationId` varchar(255) NOT NULL,
  `FloorId` varchar(255) NOT NULL,
  `CameraName` varchar(100) NOT NULL,
  `HTTPUrl` varchar(255) NOT NULL DEFAULT 'NA',
  `RTSPUrl` varchar(255) NOT NULL,
  `CameraDescription` varchar(100) NOT NULL,
  `Status` varchar(100) NOT NULL DEFAULT 'false',
  `CameraDefaultImage` mediumtext,
  `Api` varchar(255) DEFAULT 'NA',
  `Roi` json DEFAULT NULL,
  `AnalyticsConfig` json DEFAULT NULL,
  `CreatedAt` datetime NOT NULL,
  `UpdatedAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`CameraId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Cameras`
--

LOCK TABLES `Cameras` WRITE;
/*!40000 ALTER TABLE `Cameras` DISABLE KEYS */;
INSERT INTO `Cameras` VALUES ('CAM_001','WS_01','FLOOR_1','Entrance Camera1','NA','rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101','Main entrance surveillance camera','false',NULL,'NA',NULL,NULL,'2025-12-17 15:00:27','2025-12-18 10:14:28'),('CAM_002','WS_01','FLOOR_1','Entrance Camera2','NA','/home/asadel/ASADEL PROJECTS/Fall_Detection/video/3.mp4','Main entrance surveillance camera','false',NULL,'NA',NULL,NULL,'2025-12-17 15:03:41','2025-12-19 12:08:53'),('CAM_003','WS_01','FLOOR_1','Entrance Camera3','NA','/home/asadel/ASADEL PROJECTS/Fall_Detection/video/14.mp4','Main entrance surveillance camera','true',NULL,'NA',NULL,NULL,'2025-12-17 17:03:32','2025-12-19 13:02:49'),('CAM_004','WS_01','FLOOR_1','Entrance Camera4','NA','/home/asadel/ASADEL PROJECTS/Fall_Detection/video/2.mp4','Main entrance surveillance camera','false',NULL,'NA',NULL,NULL,'2025-12-17 17:04:23','2025-12-18 18:14:12');
/*!40000 ALTER TABLE `Cameras` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-12-19 16:43:15
