/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the defintions of the SMBus state machine events and
 * function declarations for the functions which generate those events
 *
 * @file smbus_event.h
 *
 */

#ifndef _SMBUS_EVENT_H_
#define _SMBUS_EVENT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"
#include "smbus_event_buffer.h"

#define E_TARGET_WRITE_IRQ                              ( 0x01 )
#define E_TARGET_READ_IRQ                               ( 0x02 )
#define E_TARGET_DATA_IRQ                               ( 0x03 )
#define E_TARGET_DONE_IRQ                               ( 0x04 )
#define E_TARGET_DESC_IRQ                               ( 0x05 )
#define E_TARGET_LOA_ERROR_IRQ                          ( 0x06 )
#define E_TARGET_PEC_ERROR_IRQ                          ( 0x07 )
#define E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ             ( 0x08 )
#define E_TARGET_RX_FIFO_ERROR_ERROR_IRQ                ( 0x09 )
#define E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ             ( 0x0A )
#define E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ            ( 0x0B )
#define E_TARGET_DESC_FIFO_ERROR_IRQ                    ( 0x0C )
#define E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ           ( 0x0D )
#define E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ          ( 0x0E )
#define E_TARGET_DESC_ERROR_IRQ                         ( 0x0F )
#define E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ         ( 0x10 )
#define E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ  ( 0x11 )
#define E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ       ( 0x12 )
#define E_CONTROLLER_WRITE_IRQ                          ( 0x13 )
#define E_CONTROLLER_READ_IRQ                           ( 0x14 )
#define E_CONTROLLER_DATA_IRQ                           ( 0x15 )
#define E_CONTROLLER_DONE_IRQ                           ( 0x16 )
#define E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ         ( 0x17 )
#define E_CONTROLLER_LOA_ERROR_IRQ                      ( 0x18 )
#define E_CONTROLLER_NACK_ERROR_IRQ                     ( 0x19 )
#define E_CONTROLLER_PEC_ERROR_IRQ                      ( 0x1A )
#define E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ    ( 0x1B )
#define E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ    ( 0x1C )
#define E_CONTROLLER_RX_FIFO_ERROR_IRQ                  ( 0x1D )
#define E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ         ( 0x1E )
#define E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ        ( 0x1F )
#define E_CONTROLLER_DESC_FIFO_ERROR_IRQ                ( 0x20 )
#define E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ       ( 0x21 )
#define E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ      ( 0x22 )
#define E_CONTROLLER_DESC_ERROR_IRQ                     ( 0x23 )
#define E_SEND_NEXT_BYTE                                ( 0x24 )
#define E_IS_PEC_REQUIRED                               ( 0x25 )
#define E_DESC_FIFO_ALMOST_EMPTY_IRQ                    ( 0x26 )

/*******************************************************************************
*
* @brief    If the instance is valid, this function will attempt to write the event
*           into the instance's event log
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
* @param    ucAnyEvent is any SMBus state machine event
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusCreateEvent( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
* with event E_IS_PEC_REQUIRED
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_IS_PEC_REQUIRED( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_SEND_NEXT_BYTE
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_SEND_NEXT_BYTE( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_WRITE_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_WRITE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DATA_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DATA_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_READ_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_READ_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DONE_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DONE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_LOA_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_LOA_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PEC_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PEC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_ERROR_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_ERROR_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DONE_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DONE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_LOA_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_LOA_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_NACK_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_NACK_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PEC_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PEC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_ERROR_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DATA_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DATA_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_DESC_FIFO_ALMOST_EMPTY_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_DESC_FIFO_ALMOST_EMPTY_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ
*
* @param    pxSMBusInstance is a pointer to the SMBus instance data
*
* @return   None
*
* @note     None.
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/*******************************************************************************
*
* @brief    Converts an event enum value to a text string for logging 
*
* @param    ucEvent is any state machine event enum value
*
* @return   A text string of the event
*
* @note     None.
*
*******************************************************************************/
char* pcEventToString( uint8_t ucEvent );

#ifdef __cplusplus
}
#endif

#endif /* _SMBUS_EVENT_H_ */
