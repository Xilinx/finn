/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the SMBus IP register offsets and bit mask defintions
 * for the SMBus driver.
 *
 * @file smbus_hardware_access.h
 *
 */

#ifndef _SMBus_HARDWARE_ACCESS_H_
#define _SMBus_HARDWARE_ACCESS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_hardware.h"

#define SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS     ( 0 )
#define SMBUS_HW_DESCRIPTOR_WRITE_FAIL        ( 1 )

typedef volatile int* SMBUS_BASE_ADDRESS_TYPE;

typedef enum SMBus_HW_Descriptor_Type
{
    DESC_TARGET_READ = 0,
    DESC_TARGET_READ_PEC,
    DESC_TARGET_WRITE_ACK,
    DESC_TARGET_WRITE_NACK,
    DESC_TARGET_WRITE_PEC,
    DESC_CONTROLLER_READ_START,
    DESC_CONTROLLER_READ_QUICK,
    DESC_CONTROLLER_READ_BYTE,
    DESC_CONTROLLER_READ_STOP,
    DESC_CONTROLLER_READ_PEC,
    DESC_CONTROLLER_WRITE_START,
    DESC_CONTROLLER_WRITE_QUICK,
    DESC_CONTROLLER_WRITE_BYTE,
    DESC_CONTROLLER_WRITE_STOP,
    DESC_CONTROLLER_WRITE_PEC,
    DESC_MAX

} SMBus_HW_Descriptor_Type;

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_VERSION
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPVersion( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_REVISION
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPRevision( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_MAGIC_NUM
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPMagicNum( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_BUILD_CONFIG_0
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadBuildConfig0( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_BUILD_CONFIG_1
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadBuildConfig1( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the GIE_ENABLE bitfield from hardware register SMBUS_REG_IRQ_GIE
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQGIEEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_IER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_ISR
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_ERR_IRQ_IER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadErrIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_ERR_IRQ_ISR
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadErrIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatus( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_SMBDAT_LOW_TIMEOUT bitfield from hardware
*           register SMBUS_REG_PHY_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusSMBDATLowTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_SMBCLK_LOW_TIMEOUT bitfield from hardware
*           register SMBUS_REG_PHY_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusSMBClkLowTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_BUS_IDLE bitfield from hardware register
*           SMBUS_REG_PHY_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusBusIdle( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_FILTER_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControl( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_ENABLE bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_DURATION bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControlDuration( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_BUS_FREE_TIME
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYBusFreetime( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the IDLE_THRESHOLD bitfield from hardware register
*           SMBUS_REG_PHY_IDLE_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYIdleThresholdIdleThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_TIMEOUT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the MIN_TIMEOUT_ENABLE bitfield from hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMinTimeoutEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the MIN_TIMEOUT_MIN bitfield from hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMinTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_TIMEOUT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMax( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the SMBCLK_FORCE_LOW bitfield from hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYResetControlSMBClkForce( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_DATA_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DATA_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDataSetupTgtDataSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DBG_STATE
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the DATA_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DATA_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_TGT_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatus( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_ACTIVE bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusActive( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_ADDRESS bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_RW bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusRW( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the PAYLOAD bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusResetBusY( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FILL_THRESHOLD bitfield from hardware register
*           SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_TGT_DBG
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDbg( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_0
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl0Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_0
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl0Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_1
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl1Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_1
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl1Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_2
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl2Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_2
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl2Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_3
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl3Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_3
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl3Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_4
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl4Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_4
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl4Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_5
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl5Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_5
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl5Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_6
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl6Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_6
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl6Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_7
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl7Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_7
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl7Address( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the DATA_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_DATA_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the START_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_START_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStartHold( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the START_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_START_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStartSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the STOP_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_STOP_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStopSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the CLK_TLOW bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CLK_TLOW
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlClkTLow( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the CLK_THIGH bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CLK_THIGH
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlClkTHigh( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the TEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the CEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the CEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the CEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_DBG_STATE
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_CTLR_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlStatusEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusResetBusy( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the PAYLOAD bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusResetBusy( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the THRESHOLD bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_CTLR_DBG
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint32_t register value read
*
* @note     None.
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_GIE
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteIRQGIEEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_IER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_ISR
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_ERR_IRQ_IER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteERRIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_ERR_IRQ_ISR
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteERRIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to ENABLE bitfield of hardware register
*           SMBUS_REG_PHY_FILTER_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYFilterControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to DURATION bitfield of hardware register
*           SMBUS_REG_PHY_FILTER_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYFilterControlDuration( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_BUS_FREE_TIME
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYBusFreetime( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to IDLE_THRESHOLD bitfield of hardware register
*           SMBUS_REG_PHY_IDLE_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYIdleThresholdIdleThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to ENABLE bitfield of hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMinTimeoutEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to TIMEOUT_MIN bitfield of hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMinTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to SMBCLK_FORCE_TIMEOUT bitfield of hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYResetControlSMBClkForceTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to SMBCLK_FORCE_LOW bitfield of hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYResetControlSMBClkForceLow( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to TGT_DATA_SETUP bitfield of hardware register
*           SMBUS_REG_PHY_TGT_DATA_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTgtDataSetupTgtDataSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_DATA_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYTgtDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_TGT_DESC_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifo( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to FIFO_ID bitfield of hardware register
*           SMBUS_REG_TGT_DESC_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifoId( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to PAYLOAD bitfield of hardware register
*           SMBUS_REG_TGT_DESC_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register
*           SMBUS_REG_TGT_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtRxFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to THRESHOLD bitfield of hardware register
*           SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Passes the ulValue to the desired TgtControl register dependent
*           on the instance value
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucInstance is SMBus instance to use
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucInstance, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Passes the ulValue to the desired TgtControl register dependent
*           on the instance value
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucInstance is SMBus instance to use
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteTgtControlAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucInstance, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_DATA_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_START_HOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStartHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_START_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStartSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_STOP_SETUP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStopSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CLK_TLOW
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlClkTLow( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CLK_THIGH
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlClkTHigh( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_PRESCALER
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_MAX
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_CTLR_CONTROL
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteCtrlControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_CTLR_DESC_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteCtrlDescFifo( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register SMBUS_REG_CTLR_DESC_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteCtrlDescFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteCtrlRxFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************
*
* @brief    Writes ulValue to FILL_THRESHOLD bitfield of hardware register
*           SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulValue value to write to the register bitfield
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusHWWriteCtrlRxFifoFillThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/* Descriptor read functions */

/*******************************************************************************
*
* @brief    Writes a data byte along with a Target Read - Read Descriptor ID
*           to transmit the data byte
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucData is the data byte being returned by the target
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusTargetReadDescriptorRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData );

/*******************************************************************************
*
* @brief    Writes a Target Read - PEC Descriptor ID to the Target Descriptor FIFO
*           To iform the IP to transmit the PEC byte it has calculated on the data
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusTargetReadDescriptorPECRead( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes a Target Write - ACK Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to transmit an ACK
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorACK( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucNoStatusCheck );

/*******************************************************************************
*
* @brief    Writes a Target Write - NACK Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to transmit a NACK
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorNACK( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes a Target Write - PEC Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to interpret the previous byte as a PEC
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorPEC( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes a Controller Write - START Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a START condition
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucDestination is the Target address to transmit to
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorStartWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination );

/*******************************************************************************
*
* @brief    Writes a Controller Write - QUICK Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a START condition followed by a STOP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucDestination is the Target address to transmit to
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorQuickWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination );

/*******************************************************************************
*
* @brief    Writes a Controller Write - BYTE Descriptor ID to the Controller Descriptor FIFO
* along with a data byte 
*           To inform the IP to transmit the data byte
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucData is the data byte to send
* @param    ucNoStatusCheck allows the function to bypass a read of the fill status of the FIFO
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorByte( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData, uint8_t ucNoStatusCheck );

/*******************************************************************************
*
* @brief    Writes a Controller Write - STOP Descriptor ID to the Controller Descriptor FIFO
* along with a data byte 
*           To inform the IP to transmit the data byte followed by a STOP condition
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucData is the final data byte to send
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorStopWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData );

/*******************************************************************************
*
* @brief    Writes a Controller Write - PEC Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit the PEC verify an ACK and then transmit a STOP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorPECWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes a Controller Read - START Descriptor ID to the Controller Descriptor FIFO
* along with Target address
*           To inform the IP to start a new READ transaction
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucDestination is the Target address to transmit to with Read bit
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorStart( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination );

/*******************************************************************************
*
* @brief    Writes a Controller Read - QUICK Descriptor ID to the Controller Descriptor FIFO
* along with Target address
*           To inform the IP to start a new READ transaction followed by a STOP
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucDestination is the Target address to transmit to with Read bit
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorQuickRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination );

/*******************************************************************************
*
* @brief    Writes a Controller Read - BYTE Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit an ACK for the previous byte received
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ucNoStatusCheck allows the function to bypass a read of the fill status of the FIFO
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorByte( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucNoStatusCheck );

/*******************************************************************************
*
* @brief    Writes a Controller Read - STOP Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a NACK followed by a STOP condition
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorStop( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Writes a Controller Read - PEC Descriptor ID to the Controller
*           Descriptor FIFO to inform the IP to transmit a NACK followed by a
*           STOP and use the last received byte to perform a PEC check
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   SMBUS_HW_DESCRIPTOR_WRITE_FAIL      - if write to descriptor fails
*           SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS   - if write to descriptor succeeds
*
* @note     None.
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorPEC( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Converts an Descriptor enum value to a text string for logging 
*
* @param    SMBus_HW_Descriptor_Type is any Descriptor enum value
*
* @return   A text string of the event
*
* @note     None.
*
*******************************************************************************/
char* pcDescriptorToString( SMBus_HW_Descriptor_Type xDescriptor );

#ifdef __cplusplus
}
#endif

#endif /* _SMBus_HARDWARE_ACCESS_H_ */
